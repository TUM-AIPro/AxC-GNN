import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Identity, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool, GINConv, SAGEConv

from quantization import IntegerQuantizer
from linear_quantized import LinearQuantized
from qconv import evaluate_prob_mask, GINConvQ, SAGEConvQ

def create_quantizer(qtype, qbits, ste, momentum, percentile, signed, sample_prop, norm=False):
    if qtype.startswith('FP'):
        return Identity
    elif qtype.startswith('INT'):
        return lambda: IntegerQuantizer(
            qbits,
            signed=signed,
            use_ste=ste,
            use_momentum=momentum,
            percentile=percentile,
            sample=sample_prop,
            norm=norm,
        )
    else:
        raise NotImplementedError


def make_quantizers(qtype, qbits, sign_input, ste, momentum, percentile, sample_prop, q_input=True, q_weight=True, q_features=True):
    layer_quantizers = {
        "inputs": create_quantizer(
            qtype if q_input else "FP32", qbits if sign_input else qbits-1, ste, momentum, percentile, sign_input, sample_prop
        ),
        "weights": create_quantizer(
            qtype if q_weight else "FP32", qbits, ste, momentum, percentile, True, sample_prop,
        ),
        "features": create_quantizer(
            qtype if q_features else "FP32", qbits, ste, momentum, percentile, True, sample_prop
        ),
    }
    mp_quantizers = {
        "message_low": create_quantizer(
            qtype, qbits, ste, momentum, percentile, True, sample_prop
        ),
        "message_high": create_quantizer(
            "FP32", 32, ste, momentum, percentile, True, sample_prop
        ),
        "update_low": create_quantizer(
            qtype, qbits, ste, momentum, percentile, True, sample_prop
        ),
        "update_high": create_quantizer(
            "FP32", 32, ste, momentum, percentile, True, sample_prop
        ),
        "aggregate_low": create_quantizer(
            qtype, qbits, ste, momentum, percentile, True, sample_prop
        ),
        "aggregate_high": create_quantizer(
            "FP32", 32, ste, momentum, percentile, True, sample_prop
        ),
    }
    return layer_quantizers, mp_quantizers


class ResettableSequential(Sequential):
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()


class GIN(torch.nn.Module):
    def __init__(
        self,
        dataset,
        num_layers,
        hidden,
        qtype,
        qbits,
        ste,
        momentum,
        percentile,
        sample_prop,
        q_input=True,
        q_weight=True,
        q_features=True,
        apx=0,
    ):
        super(GIN, self).__init__()

        gin_layer = GINConvQ

        lq, mq = make_quantizers(
            qtype,
            qbits,
            False,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
            q_input=q_input,
            q_weight=q_weight,
            q_features=q_features
        )
        lq_signed, _ = make_quantizers(
            qtype,
            qbits,
            True,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
            q_input=q_input,
            q_weight=q_weight,
            q_features=q_features
        )

        if qbits == 32:
            self.conv1 = GINConv(
                ResettableSequential(
                    Linear(dataset.num_features, hidden, bias=False),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden, bias=False),
                    ReLU(),
                ),
                train_eps=False,
            )
            self.convs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self.convs.append(
                    GINConv(
                        ResettableSequential(
                            Linear(hidden, hidden, bias=False),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden, bias=False),
                            ReLU(),
                        ),
                        train_eps=False,
                    )
                )

            self.lin1 = Linear(hidden, hidden, bias=False)
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            self.conv1 = gin_layer(
                ResettableSequential(
                    Linear(dataset.num_features, hidden, bias=False),
                    BN(hidden),
                    ReLU(),
                    LinearQuantized(hidden, hidden, layer_quantizers=lq, apx=apx, bias=False),
                    ReLU(),
                ),
                train_eps=False,
                mp_quantizers=mq,
            )
            self.convs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self.convs.append(
                    gin_layer(
                        ResettableSequential(
                            LinearQuantized(hidden, hidden, layer_quantizers=lq_signed, apx=apx, bias=False),
                            BN(hidden),
                            ReLU(),
                            LinearQuantized(hidden, hidden, layer_quantizers=lq, apx=apx, bias=False),
                            ReLU(),
                        ),
                        train_eps=False,
                        mp_quantizers=mq,
                    )
                )

            self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed, apx=apx, bias=False)
            self.lin2 = LinearQuantized(hidden, dataset.num_classes, layer_quantizers=lq, apx=apx)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        if hasattr(data, "prob_mask") and data.prob_mask is not None:
            mask = evaluate_prob_mask(data)
        else:
            mask = None
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index, mask)
        for conv in self.convs:
            x = conv(x, edge_index, mask)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class CitGIN(torch.nn.Module):
    def __init__(
        self,
        dataset,
        hidden,
        qtype,
        qbits,
        ste,
        momentum,
        percentile,
        sample_prop,
        q_input=True,
        q_weight=True,
        q_features=True,
        apx=0,
    ):
        super(CitGIN, self).__init__()

        if qbits == 32:
            self.conv1 = GINConv(
                ResettableSequential(
                    Linear(dataset.num_features, hidden, bias=False),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden, bias=False),
                    ReLU(),
                ),
                train_eps=False,
            )
            self.conv2 = GINConv(
                ResettableSequential(
                    Linear(hidden, hidden, bias=False),
                    ReLU(),
                    Linear(hidden, dataset.num_classes, bias=False),
                    ReLU(),
                ),
                train_eps=False,
            )
        else:
            gin_layer = GINConvQ
            lq, mq = make_quantizers(
                qtype,
                qbits,
                False,
                ste=ste,
                momentum=momentum,
                percentile=percentile,
                sample_prop=sample_prop,
                q_input=q_input,
                q_weight=q_weight,
                q_features=q_features
            )
            lq_signed, _ = make_quantizers(
                qtype,
                qbits,
                True,
                ste=ste,
                momentum=momentum,
                percentile=percentile,
                sample_prop=sample_prop,
                q_input=q_input,
                q_weight=q_weight,
                q_features=q_features
            )
            self.conv1 = gin_layer(
                ResettableSequential(
                    LinearQuantized(dataset.num_features, hidden, layer_quantizers=lq, apx=apx, bias=False),
                    BN(hidden),
                    ReLU(),
                    LinearQuantized(hidden, hidden, layer_quantizers=lq, apx=apx, bias=False),
                    ReLU(),
                ),
                train_eps=False,
                mp_quantizers=mq,
            )
            self.conv2 = gin_layer(
                ResettableSequential(
                    LinearQuantized(hidden, hidden, layer_quantizers=lq_signed, apx=apx, bias=False),
                    ReLU(),
                    LinearQuantized(hidden, dataset.num_classes, layer_quantizers=lq, apx=apx, bias=False),
                    ReLU(),
                ),
                train_eps=False,
                mp_quantizers=mq,
            )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        if hasattr(data, "prob_mask") and data.prob_mask is not None:
            mask = evaluate_prob_mask(data)
        else:
            mask = None

        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, mask)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, mask)

        return x

class SAGE(torch.nn.Module):
    def __init__(
        self,
        dataset,
        num_layers,
        hidden,
        qtype,
        qbits,
        ste,
        momentum,
        percentile,
        sample_prop,
        q_input=True,
        q_weight=True,
        q_features=True,
        apx=0,
    ):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.activs = torch.nn.ModuleList()
        if qbits == 32:
            self.conv1 = SAGEConv(
                    dataset.num_features,
                    hidden,
                    normalize=False,
                    bias=False,
                )
            for i in range(num_layers - 2):
                self.convs.append(
                    SAGEConv(
                        hidden,
                        hidden,
                        normalize=False,
                        bias=False,
                    )
                )
                self.activs.append(ReLU())
            self.convs.append(
                    SAGEConv(
                        hidden,
                        hidden,
                        normalize=False,
                        bias=False,
                    )
            )
            self.lin1 = Linear(hidden, hidden, bias=False)
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            sage_conv = SAGEConvQ
            lq, mq = make_quantizers(
                qtype,
                qbits,
                False,
                ste=ste,
                momentum=momentum,
                percentile=percentile,
                sample_prop=sample_prop,
                q_input=q_input,
                q_weight=q_weight,
                q_features=q_features
            )
            lq_signed, _ = make_quantizers(
                qtype,
                qbits,
                True,
                ste=ste,
                momentum=momentum,
                percentile=percentile,
                sample_prop=sample_prop,
                q_input=q_input,
                q_weight=q_weight,
                q_features=q_features
            )
            self.conv1 = SAGEConv(
                    dataset.num_features,
                    hidden,
                    normalize=False,
                    bias=False,
                )
            for i in range(num_layers - 2):
                self.convs.append(
                    sage_conv(
                        hidden,
                        hidden,
                        layer_quantizers=lq,
                        mp_quantizers=mq,
                        apx=apx,
                    )
                )
                self.activs.append(ReLU())
            self.convs.append(
                sage_conv(
                    hidden,
                    hidden,
                    layer_quantizers=lq,
                    mp_quantizers=mq,
                    apx=apx,
                )
            )
            self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed, apx=apx, bias=False)
            self.lin2 = LinearQuantized(hidden, dataset.num_classes, layer_quantizers=lq, apx=apx)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        if hasattr(data, "prob_mask") and data.prob_mask is not None:
            mask = evaluate_prob_mask(data)
        else:
            mask = None

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = conv(x, edge_index, mask)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class CitSAGE(torch.nn.Module):
    def __init__(
        self,
        dataset,
        hidden,
        qtype,
        qbits,
        ste,
        momentum,
        percentile,
        sample_prop,
        q_input=True,
        q_weight=True,
        q_features=True,
        apx=0,
    ):
        super(CitSAGE, self).__init__()
        if qbits == 32:
            self.conv1 = SAGEConv(
                dataset.num_features,
                hidden,
                normalize=False,
                bias=False,
            )
            self.conv2 = SAGEConv(
                hidden,
                dataset.num_classes,
                normalize=False,
                bias=True,
            )
        else:
            sage_conv = SAGEConvQ
            lq, mq = make_quantizers(
                qtype,
                qbits,
                False,
                ste=ste,
                momentum=momentum,
                percentile=percentile,
                sample_prop=sample_prop,
                q_input=q_input,
                q_weight=q_weight,
                q_features=q_features
            )
            self.conv1 = sage_conv(
                    dataset.num_features,
                    hidden,
                    layer_quantizers=lq,
                    mp_quantizers=mq,
                    apx=apx,
                )
            self.conv2 = sage_conv(
                    hidden,
                    dataset.num_classes,
                    layer_quantizers=lq,
                    mp_quantizers=mq,
                    apx=apx,
                    bias=True,
                )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        if hasattr(data, "prob_mask") and data.prob_mask is not None:
            mask = evaluate_prob_mask(data)
        else:
            mask = None

        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index, mask))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, mask)
        return x