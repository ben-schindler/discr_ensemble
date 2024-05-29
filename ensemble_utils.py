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

class split_to_discriminators(torch.autograd.Function):
    """expand sample as an input to multiple discriminators, on backward, aggregate """

    @staticmethod
    def forward(ctx, tensor, n_of_discr):
        """
        Just forward the input.
        """
        ctx.gen_attached = not tensor.is_leaf,
        ctx.n_of_discr =  n_of_discr
        return tensor.expand([n_of_discr] + [-1]*tensor.dim())

    @staticmethod
    def backward(ctx, tensor_grad):
        """
        Random weight the gradients according to normal distribuation.
        """
        return tensor_grad.sum(dim=0), None

class weight_rand_uniform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        n_of_discr = tensor_grad.shape[0]
        w = F.softmax(torch.rand(n_of_discr, device=tensor_grad.device), dim=0)
        view_shape = [-1] + [1] * (tensor_grad.dim() - 1)
        tensor_grad = tensor_grad * w.view(view_shape)
        return tensor_grad

class weight_rand_normal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        n_of_discr = tensor_grad.shape[0]
        w = F.softmax(torch.randn(n_of_discr, device=tensor_grad.device), dim=0)
        view_shape = [-1] + [1] * (tensor_grad.dim() - 1)
        tensor_grad = tensor_grad * w.view(view_shape)
        return tensor_grad

class weight_rand_bernoulli(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        n_of_discr = tensor_grad.shape[0]
        w = torch.bernoulli(torch.tensor([0.5], device=tensor_grad.device).repeat(n_of_discr))
        if w.sum() == 0.:
            w[torch.randint(high=n_of_discr, size=[1])] = 1.
        else:
            w.div(w.sum())
        view_shape = [-1] + [1] * (tensor_grad.dim() - 1)
        tensor_grad = tensor_grad * w.view(view_shape)
        return tensor_grad

class weight_by_predicts(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, lambda_var, discr_outputs):
        """
        Forward the
        """
        ctx.lambda_var = lambda_var
        ctx.discr_outputs = discr_outputs
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        if ctx.discr_outputs._version != 1:
            raise RuntimeError("Tensor of discr_output have to be changed exactly one (at the end of forward path)")
        feature_dims = tensor_grad.dim() - 2
        no_of_discr = ctx.discr_outputs.shape[1]
        w = F.softmax(-ctx.discr_outputs * ctx.lambda_var, dim=1)
        tensor_grad = tensor_grad * w.movedim(0, -1).view([no_of_discr, -1] + [1] * feature_dims)
        return tensor_grad, None, None

class weight_by_predict_logits(torch.autograd.Function):
    '''use this function instead of weight_by_predicts, when your discriminator outputs logits
    instead of the vanilla [0,1]-Classification, '''

    @staticmethod
    def forward(ctx, tensor, lambda_var, discr_outputs):
        """
        Forward the
        """
        ctx.lambda_var = lambda_var
        ctx.discr_outputs = discr_outputs
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        if ctx.discr_outputs._version != 1:
            raise RuntimeError("Tensor of discr_output have to be changed exactly one (at the end of forward path)")
        feature_dims = tensor_grad.dim() - 2
        no_of_discr = ctx.discr_outputs.shape[1]
        w = F.softmax(-F.sigmoid(discr_outputs) * ctx.lambda_var, dim=1)
        tensor_grad = tensor_grad * w.movedim(0, -1).view([no_of_discr, -1] + [1] * feature_dims)
        return tensor_grad, None, None

class weight_fixed(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, weights):
        if weights.shape[0] != tensor.shape[0]:
            raise ValueError('Number of weightings must equal the number of discriminators.')
        ctx.weights = weights.to(tensor.device)
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        n_of_discr = tensor_grad.shape[0]
        w = ctx.weights
        view_shape = [-1] + [1] * (tensor_grad.dim() - 1)
        tensor_grad = tensor_grad * w.view(view_shape)
        return tensor_grad, None

class gradient_normalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        n_of_discr = tensor_grad.shape[0]
        overall_mean = tensor_grad.flatten().abs().mean()
        w = overall_mean / tensor_grad.flatten(1).abs().mean(dim=1)
        view_shape = [-1] + [1] * (tensor_grad.dim() - 1)
        tensor_grad = tensor_grad * w.view(view_shape)
        return tensor_grad

def reduce_output(x):
    return reduce_output_function.apply(x)

def get_gradient_weighting(weighting: str, fixed_weights=None):
    if weighting not in ["ew", "soft", "rand_uniform", "rand_normal", "rand_bernoulli", "fixed", "soft_logits"]:
        raise ValueError('Weighting must be in ["ew", "soft", "rand_uniform", "rand_normal", "rand_bernoulli", "fixed", "soft_logits"].')
    if weighting == "ew":
        return lambda x: x
    if weighting == "rand_uniform":
        return lambda x: weight_rand_uniform.apply(x)
    if weighting == "rand_normal":
        return lambda x: weight_rand_normal.apply(x)
    if weighting == "rand_bernoulli":
        return lambda x: weight_rand_bernoulli.apply(x)
    if weighting == "soft":
        return lambda x, lambda_var, discr_outputs: weight_by_predicts.apply(x, lambda_var, discr_outputs)
    if weighting == "soft_logits":
        return lambda x, lambda_var, discr_outputs: weight_by_predict_logits.apply(x, lambda_var, discr_outputs)
    if weighting == "fixed":
        if not isinstance(fixed_weights, list) or not all(isinstance(w, float) for w in fixed_weights):
            raise ValueError('Fixed Weighting must be a list of floats (e.g. ([0.30, 0.30, 0,40])).')
        else:
            fixed_weights = torch.tensor(fixed_weights)
            return lambda x: weight_fixed.apply(x, fixed_weights)


'''
def soft_weighting_autograd(*args):
    return soft_weighting_function.apply(*args)
    

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
                w = F.softmax(torch.rand(ctx.n_of_discr, device=tensor_grad.device), dim=0)
                view_shape = [-1] + [1] * (tensor_grad.dim()-1)
                tensor_grad = tensor_grad *  w.view(view_shape)
            if ctx.weighting == "rand_normal":
                w = F.softmax(torch.randn(ctx.n_of_discr, device=tensor_grad.device), dim=0)
                view_shape = [-1] + [1] * (tensor_grad.dim()-1)
                tensor_grad = tensor_grad *  w.view(view_shape)
            if ctx.weighting == "rand_bernoulli":
                w = torch.bernoulli(torch.tensor([0.5], device=tensor_grad.device).repeat(ctx.n_of_discr))
                if w.sum() == 0.:
                    w[torch.randint(high=ctx.n_of_discr, size=[1])] = 1.
                else:
                    w.div(w.sum())
                view_shape = [-1] + [1] * (tensor_grad.dim() - 1)
                tensor_grad = tensor_grad * w.view(view_shape)
        return tensor_grad.sum(dim=0), None, None





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
            feature_dims = tensor_grad.dim() - 2
            no_of_discr = ctx.discr_outputs.shape[1]
            w = F.softmax(ctx.discr_outputs * ctx.lambda_var, dim=1)
            tensor_grad = tensor_grad * w.movedim(0,-1).view([no_of_discr,-1] + [1] * feature_dims)
        return tensor_grad, None, None
        
'''



