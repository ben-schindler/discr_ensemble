"""
Discriminator Ensemble Utilities
"""

import torch
import torch.nn.functional as F


class reduce_output_function(torch.autograd.Function):
    """Reduce multiple discriminator outputs to a single output"""

    @staticmethod
    def forward(ctx, discr_outputs):
        """
        Save Discriminator outputs and forward the mean.
        """
        ctx.save_for_backward(discr_outputs)
        return torch.mean(discr_outputs, dim=-1, keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        """
        During Backpropagation multiply the Gradients with the number of Discriminators,
        to preserve the gradient magnitude compared to a single discriminator.
        """
        discr_outputs, = ctx.saved_tensors
        return grad_output.repeat(discr_outputs.shape[0])

class split_and_weight(torch.autograd.Function):
    """expand sample as an input to multiple discriminators,
    do a weighting on backward if the sample is attached to a generator (no leaf)"""

    @staticmethod
    def forward(ctx, tensor, n_of_discr, weighting):
        """
        Just forward the input.
        """
        ctx.gen_attached = not tensor.is_leaf,
        ctx.n_of_discr =  n_of_discr
        ctx.weighting = weighting
        return tensor.expand([n_of_discr] + [-1]*tensor.dim())

    @staticmethod
    def backward(ctx, tensor_grad):
        """
        Random weight the gradients according to normal distribuation.
        """
        if ctx.gen_attached:
            if ctx.weighting == "rand_uniform":
                w = F.softmax(torch.rand(ctx.n_of_discr), dim=0)
                view_shape = [-1] + [1] * (tensor_grad.dim()-1)
                tensor_grad = tensor_grad *  w.view(view_shape)
            if ctx.weighting == "rand_normal":
                w = F.softmax(torch.randn(ctx.n_of_discr), dim=0)
                view_shape = [-1] + [1] * (tensor_grad.dim()-1)
                tensor_grad = tensor_grad *  w.view(view_shape)
            if ctx.weighting == "rand_bernoulli":
                w = torch.bernoulli(torch.tensor([0.5]).repeat(ctx.n_of_discr))
                if w.sum() == 0.:
                    w[torch.randint(high=ctx.n_of_discr, size=[1])] = 1.
                else:
                    w.div(w.sum())
                view_shape = [-1] + [1] * (tensor_grad.dim() - 1)
                tensor_grad = tensor_grad * w.view(view_shape)
        return tensor_grad.sum(dim=0), None, None


class split_and_soft_weight(torch.autograd.Function):
    """expand sample as an input to multiple discriminators,
    do a weighting on backward if the sample is attached to a generator (no leaf)"""

    @staticmethod
    def forward(ctx, tensor, n_of_discr, weighting):
        """
        Just forward the input.
        """
        ctx.gen_attached = not tensor.is_leaf,
        ctx.n_of_discr =  n_of_discr
        ctx.weighting = weighting
        return tensor.expand([n_of_discr] + [-1]*tensor.dim())

    @staticmethod
    def backward(ctx, tensor_grad):
        """
        Random weight the gradients according to normal distribuation.
        """
        if ctx.gen_attached:
            if ctx.weighting == "rand_uniform":
                w = F.softmax(torch.rand(ctx.n_of_discr), dim=0)
                view_shape = [-1] + [1] * (tensor_grad.dim()-1)
                tensor_grad = tensor_grad *  w.view(view_shape)
            if ctx.weighting == "rand_normal":
                w = F.softmax(torch.randn(ctx.n_of_discr), dim=0)
                view_shape = [-1] + [1] * (tensor_grad.dim()-1)
                tensor_grad = tensor_grad *  w.view(view_shape)
            if ctx.weighting == "rand_bernoulli":
                w = torch.bernoulli(torch.tensor([0.5]).repeat(ctx.n_of_discr))
                if w.sum() == 0.:
                    w[torch.randint(high=ctx.n_of_discr, size=[1])] = 1.
                else:
                    w.div(w.sum())
                view_shape = [-1] + [1] * (tensor_grad.dim() - 1)
                tensor_grad = tensor_grad * w.view(view_shape)
        return tensor_grad.sum(dim=0), None, None


def soft_weighting(grad, predictions, lambda_var):
    "weighting according to discriminator predictions, GMAN paper"
    w = F.softmax(predictions * lambda_var, dim=0)
    view_shape = [-1] + [1] * (grad.dim() - 1)
    grad = grad * w.view(view_shape)
    return grad



class soft_weighting_function(torch.autograd.Function):
    """Reduce multiple discriminator outputs to a single output"""

    @staticmethod
    def forward(ctx, tensor, lambda_var, discr_outputs):
        """
        Forward the
        """
        ctx.gen_attached = not tensor.is_leaf,
        ctx.lambda_var = lambda_var
        ctx.discr_outputs = discr_outputs
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        """
        During Backpropagation multiply the Gradients with the number of Discriminators,
        to preserve the gradient magnitude compared to a single discriminator.
        """
        if ctx.gen_attached:
            if ctx.discr_outputs._version !=1:
                raise RuntimeError("Tensor of discr_output have to be changed exactly one (at the end of forward path)")
            w = F.softmax(ctx.discr_outputs * ctx.lambda_var, dim=1)
            tensor_grad = tensor_grad * w.movedim(0,-1).view(ctx.discr_outputs.shape[1],-1,1)
        return tensor_grad, None, None

def reduce_output(x):
    return reduce_output_function.apply(x)

def soft_weighting_autograd(*args):
    return soft_weighting_function.apply(*args)


