import inspect
from collections import OrderedDict
from typing import Union, Tuple
import torch
from torch import Tensor
import torch_scatter
from torch.nn import Module, ModuleDict, Sequential, Parameter, ReLU, Identity, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.utils import (
    add_remaining_self_loops,
    remove_self_loops,
    add_self_loops,
    softmax,
)
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size, SparseTensor
from linear_quantized import LinearQuantized, apxLinearF

msg_special_args = set(
    [
        "edge_index",
        "edge_index_i",
        "edge_index_j",
        "size",
        "size_i",
        "size_j",
    ]
)

aggr_special_args = set(
    [
        "index",
        "dim_size",
    ]
)

update_special_args = set([])

def evaluate_prob_mask(data):
    return torch.bernoulli(data.prob_mask).to(torch.bool)

def scatter_(name, src, index, dim=0, dim_size=None):
    assert name in ["add", "mean", "min", "max"]

    op = getattr(torch_scatter, "scatter_{}".format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out

    if name == "max":
        out[out < -10000] = 0
    elif name == "min":
        out[out > 10000] = 0

    return out

REQUIRED_QUANTIZER_KEYS = [
    "message_low",
    "message_high",
    "update_low",
    "update_high",
    "aggregate_low",
    "aggregate_high",
]


class MessagePassingQ(Module):
    # https://github.com/camlsys/degree-quant
    # S. A. Tailor, J. Fern ́andez-Marqu ́es, and N. D. Lane, “Degree-
    # quant: Quantization-aware training for graph neural networks,” ArXiv,
    # vol. abs/2008.05000, 2020
    def __init__(
        self,
        aggr="add",
        flow="source_to_target",
        node_dim=0,
        mp_quantizers=None,
    ):
        super(MessagePassingQ, self).__init__()

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max"]

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__args__ = set().union(msg_args, aggr_args, update_args)

        assert mp_quantizers is not None
        self.mp_quant_fns = mp_quantizers

    def reset_parameters(self):
        self.mp_quantizers = ModuleDict()
        for key in REQUIRED_QUANTIZER_KEYS:
            self.mp_quantizers[key] = self.mp_quant_fns[key]()

    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (
                    f"Encountered node tensor with size "
                    f"{tensor.size(self.node_dim)} in dimension {self.node_dim}, "
                    f"but expected size {size[index]}."
                )
            )

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        out["edge_index"] = edge_index
        out["edge_index_i"] = edge_index[i]
        out["edge_index_j"] = edge_index[j]
        out["size"] = size
        out["size_i"] = size[i]
        out["size_j"] = size[j]

        # Add special aggregate arguments.
        out["index"] = out["edge_index_i"]
        out["dim_size"] = out["size_i"]

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"Required parameter {key} is empty.")
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, mask, size=None, **kwargs):
        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2
        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        msg = self.message(**msg_kwargs)
        if self.training:
            edge_mask = torch.index_select(mask, 0, edge_index[0])
            out = torch.empty_like(msg)
            out[edge_mask] = self.mp_quantizers["message_high"](msg[edge_mask])
            out[~edge_mask] = self.mp_quantizers["message_low"](msg[~edge_mask])[0]
        else:
            out = self.mp_quantizers["message_low"](msg)[0]

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        aggrs = self.aggregate(out, **aggr_kwargs)
        if self.training:
            out = torch.empty_like(aggrs)
            out[mask] = self.mp_quantizers["aggregate_high"](aggrs[mask])
            out[~mask] = self.mp_quantizers["aggregate_low"](aggrs[~mask])[0]
        else:
            out = self.mp_quantizers["aggregate_low"](aggrs)[0]

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        updates = self.update(out, **update_kwargs)
        if self.training:
            out = torch.empty_like(updates)
            out[mask] = self.mp_quantizers["update_high"](updates[mask])
            out[~mask] = self.mp_quantizers["update_low"](updates[~mask])[0]
        else:
            out = self.mp_quantizers["update_low"](updates)[0]

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

    def update(self, inputs):  # pragma: no cover
        return inputs

REQUIRED_GCN_KEYS = [
    "weights",
    "inputs",
    "inputs_high",
    "features",
    "features_high",
    "norm",
]


class GCNConvQ(MessagePassingQ):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved=False,
        cached=False,
        bias=True,
        normalize=True,
        layer_quantizers=None,
        mp_quantizers=None,
        apx=0,
        **kwargs,
    ):
        super(GCNConvQ, self).__init__(
            aggr="add", mp_quantizers=mp_quantizers, **kwargs
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.linear_function = apxLinearF.apply
        self.apx = apx
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

        # create quantization modules for this layer
        self.layer_quantizers = ModuleDict()
        for key in REQUIRED_GCN_KEYS:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, mask, edge_weight=None):
        # quantizing input
        if self.training:
            x_q = torch.empty_like(x)
            x_q[mask] = self.layer_quantizers["inputs_high"](x[mask])
            x_q[~mask], x_scale, x_zp = self.layer_quantizers["inputs"](x[~mask])
        else:
            x_q, x_scale, x_zp = self.layer_quantizers["inputs"](x)

        # quantizing layer weights
        w_q, w_scale, w_zp = self.layer_quantizers["weights"](self.weight)
        if self.training:
            x = torch.empty((x_q.shape[0], w_q.shape[1])).to(x_q.device)
            x_tmp = self.linear_function(x_q, w_q, x_scale, x_zp, w_scale, w_zp, self.apx)
            x[mask] = self.layer_quantizers["features_high"](x_tmp[mask])
            x[~mask] = self.layer_quantizers["features"](x_tmp[~mask])[0]
        else:
            x_tmp = self.linear_function(x_q, w_q, x_scale, x_zp, w_scale, w_zp, self.apx)
            x = self.layer_quantizers["features"](x_tmp)[0]

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(
                        self.cached_num_edges, edge_index.size(1)
                    )
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(
                    edge_index,
                    x.size(self.node_dim),
                    edge_weight,
                    self.improved,
                    x.dtype,
                )
            else:
                norm = edge_weight
            norm = self.layer_quantizers["norm"](norm)[0]
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm, mask=mask)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


REQUIRED_GAT_KEYS = [
    "weights",
    "inputs",
    "inputs_high",
    "features",
    "features_high",
    "attention",
    "alpha",
    "alpha_high",
]


class GATConvQ(MessagePassingQ):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        layer_quantizers=None,
        mp_quantizers=None,
        apx=0,
        **kwargs,
    ):
        super(GATConvQ, self).__init__(
            aggr="add", mp_quantizers=mp_quantizers, **kwargs
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.linear_function = apxLinearF.apply
        self.apx = apx
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

        # create quantization modules for this layer
        self.layer_quantizers = ModuleDict()
        for key in REQUIRED_GAT_KEYS:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    def forward(self, x, edge_index, mask, size=None):
        # quantizing input
        if self.training:
            x_q = torch.empty_like(x)
            x_q[mask] = self.layer_quantizers["inputs_high"](x[mask])
            x_q[~mask], x_scale, x_zp = self.layer_quantizers["inputs"](x[~mask])
        else:
            x_q, x_scale, x_zp = self.layer_quantizers["inputs"](x)

        # quantizing layer weights
        w_q, w_scale, w_zp = self.layer_quantizers["weights"](self.weight)

        if size is None and torch.is_tensor(x_q):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x_q.size(0))

        if torch.is_tensor(x_q):
            if self.training:
                x = torch.empty((x_q.shape[0], w_q.shape[1])).to(x_q.device)
                x_tmp = self.linear_function(x_q, w_q, x_scale, x_zp, w_scale, w_zp, self.apx)
                # x_tmp = torch.mm(x_q, w_q)
                x[mask] = self.layer_quantizers["features_high"](x_tmp[mask])
                x[~mask] = self.layer_quantizers["features"](x_tmp[~mask])[0]
            else:
                x_tmp = self.linear_function(x_q, w_q, x_scale, x_zp, w_scale, w_zp, self.apx)
                # x_tmp = torch.mm(x_q, w_q)
                x = self.layer_quantizers["features"](x_tmp)[0]

            x_q = x
        else:
            x = (
                None if x_q[0] is None else torch.matmul(x_q[0], w_q),
                None if x_q[1] is None else torch.matmul(x_q[1], w_q),
            )
            if self.training:
                x0_q = None
                if x[0] is not None:
                    x0_q = torch.empty_like(x[0])
                    x0_q[mask] = self.layer_quantizers["features_high"](x[0][mask])
                    x0_q[~mask] = self.layer_quantizers["features"](x[0][~mask])[0]

                x1_q = None
                if x[1] is not None:
                    x1_q = torch.empty_like(x[1])
                    x1_q[mask] = self.layer_quantizers["features_high"](x[1][mask])
                    x1_q[~mask] = self.layer_quantizers["features"](x[1][~mask])[0]

                x_q = (x0_q, x1_q)

            else:
                x_q = (
                    None
                    if x[0] is None
                    else self.layer_quantizers["features"](x[0])[0],
                    None
                    if x[1] is None
                    else self.layer_quantizers["features"](x[1])[0],
                )

        edge_mask = torch.index_select(mask, 0, edge_index[0])
        return self.propagate(
            edge_index, size=size, x=x_q, mask=mask, edge_mask=edge_mask
        )

    def message(self, edge_index_i, x_i, x_j, size_i, edge_mask):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        att = self.layer_quantizers["attention"](self.att)[0]

        if x_i is None:
            alpha = (x_j * att[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * att).sum(dim=-1)

        if self.training:
            alpha_tmp = torch.empty_like(alpha)
            alpha_tmp[edge_mask] = self.layer_quantizers["alpha_high"](alpha[edge_mask])
            alpha_tmp[~edge_mask] = self.layer_quantizers["alpha"](alpha[~edge_mask])[0]
            alpha = alpha_tmp
        else:
            alpha = self.layer_quantizers["alpha"](alpha)[0]

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, None, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
       
class GINConvQ(MessagePassingQ):
    def __init__(self, nn, eps=0, train_eps=False, mp_quantizers=None, **kwargs):
        super(GINConvQ, self).__init__(
            aggr="add", mp_quantizers=mp_quantizers, **kwargs
        )
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, mask):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=x, mask=mask)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return self.nn((1 + self.eps) * x + aggr_out)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)

class ResettableSequential(Sequential):
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()

class SAGEConvQ(MessagePassingQ):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        bias: bool = False,
        normalize: bool = False,
        mp_quantizers=None,
        apx=0,
        layer_quantizers=None,
        **kwargs
    ):

        super(SAGEConvQ, self).__init__(aggr="mean", mp_quantizers=mp_quantizers, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = ResettableSequential(
            BN(in_channels[0], momentum=0.999),
            LinearQuantized(in_channels[0], out_channels, layer_quantizers, apx=apx, bias=bias)
        )
        self.lin_r = ResettableSequential(
            BN(in_channels[1], momentum=0.999),
            LinearQuantized(in_channels[1], out_channels, layer_quantizers, apx=apx, bias=False)
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, mask: Tensor, size: Size = None
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, mask=mask, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
    
class GINConvQnpp(MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, mp_quantizers=None, **kwargs):
        super(GINConvQnpp, self).__init__(
            aggr="add", mp_quantizers=mp_quantizers, **kwargs
        )
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class SAGEConvQnpp(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        bias: bool = False,
        normalize: bool = False,
        mp_quantizers=None,
        apx=0,
        layer_quantizers=None,
        **kwargs
    ):

        super(SAGEConvQnpp, self).__init__(aggr="sum", mp_quantizers=mp_quantizers, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = ResettableSequential(
            BN(in_channels[0], momentum=0.999),
            LinearQuantized(in_channels[0], out_channels, layer_quantizers, apx=apx, bias=bias)
        )
        self.lin_r = ResettableSequential(
            BN(in_channels[1], momentum=0.999),
            LinearQuantized(in_channels[1], out_channels, layer_quantizers, apx=apx, bias=False)
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )