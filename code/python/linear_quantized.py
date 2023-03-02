import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import apxop

class apxLinear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, input_scale, i_zero_point, w_scale, w_zero_point, apx):
        ctx.save_for_backward(input, weight, bias)

        input.div_(input_scale).add_(i_zero_point)
        weight.div_(w_scale).add_(w_zero_point)
        input = input.to(dtype=torch.int)
        weight = weight.to(dtype=torch.int)

        output = apxop.mm(input, weight.t().contiguous(), apx).float().mul_(input_scale * w_scale)

        if bias is not None:
            output.add(bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class LinearQuantized(nn.Linear):
    def __init__(
        self, in_features, out_features, layer_quantizers, apx=0, bias=True
    ):
        self.layer_quant_fns = layer_quantizers
        self.apx = apx
        super(LinearQuantized, self).__init__(in_features, out_features, bias)
        self.linear_function = apxLinear.apply

    def reset_parameters(self):
        super().reset_parameters()
        self.layer_quant = nn.ModuleDict()
        for key in ["inputs", "features", "weights"]:
            self.layer_quant[key] = self.layer_quant_fns[key]()

    def forward(self, input):
        input_q, input_scale, i_zero_point = self.layer_quant["inputs"](input)
        w_q, w_scale, w_zero_point = self.layer_quant["weights"](self.weight)
        out = self.linear_function(input_q, w_q, self.bias, input_scale, i_zero_point, w_scale, w_zero_point, self.apx)
        out, _, _ = self.layer_quant["features"](out)

        return out
