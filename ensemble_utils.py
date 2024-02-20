"""
Discriminator Ensemble Utilities
"""

import torch


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

def reduce_output(x):
    return reduce_output_function.apply(x)

