"""
IGN_Discriminators.

Discriminator Models for Impedance Spectra.
date: 31.05.2022
author: Benjamin Schindler
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

import json
import copy
import warnings
from typing import Union
from easydict import EasyDict as edict

from ensemble_utils import reduce_output, get_gradient_weighting


#from functorch import combine_state_for_ensemble
#from functorch import vmap

'''
class DiscriminatorGroup(nn.Module):
    """
    A Group of vectorized Discriminators with the exact same architector
    """

    def __init__(self, discr_class: type, multiplier: int = 1, config=None, *args, **kwargs):
        super().__init__()

        if config is not None:
            args = [config] + list(args)
        group = [discr_class(*args, **kwargs) for _ in range(multiplier)]
        self.fmodel, self.params, self.buffers = combine_state_for_ensemble(group)

    def forward(self,x):
        return vmap(self.fmodel, (0,0,None), -2)(self.params, self.buffers, x).squeeze(-1)

'''

class DiscriminatorEnsemble(nn.Module):
    """
    Discriminator MLP-Discriminator.

    This implements a
    config-arguments:
    - discriminators: number of input features
    """

    def __init__(self, discr_class: Union[type, list[type]], config: Union[dict, list] = None, multiplier: int = 1,
                 weighting: str = 'ew', lambda_var=1, fixed_weights=None, split_batch=False, isStudioGAN=False, *args, **kwargs):
        super().__init__()
        self.isStudioGAN = isStudioGAN

        # convert discr_class to a list if single type is given:
        if not isinstance(discr_class, list):
            discr_class = [discr_class]

        # convert config to a list if only one configuration is given:
        if not isinstance(config, list):
            config = [config]

        if config is not None and len(discr_class) != len(config):
            raise ValueError("Number of given discr_classes must equal the number of configs")

        if weighting not in ["ew", "soft", "rand_uniform", "rand_normal", "rand_bernoulli", "fixed"]:
            raise ValueError('Weighting must be in ["ew", "soft", "rand_uniform", "rand_normal", "rand_bernoulli", "fixed"]')

        if weighting == "fixed":
            if not isinstance(fixed_weights, list) or not all(isinstance(w, float) for w in fixed_weights):
                raise ValueError('Fixed Weighting must be a list of floats (e.g. ([0.30, 0.30, 0,40])).')

        self.n_of_groups = len(discr_class)
        self.group_size = multiplier
        self.n_of_discr = len(discr_class) * multiplier
        self.lambda_var = torch.tensor(lambda_var)
        self.weighting_method = weighting
        self.fixed_weights = fixed_weights
        self.weight = get_gradient_weighting(weighting, fixed_weights=self.fixed_weights)
        self.split_batch = split_batch

        self.discriminators = nn.ModuleList()

        for idx, module_class in enumerate(discr_class):

            for disc_idx in range(multiplier):
                if self.isStudioGAN: # -> don't use config, but pass arguments directly
                    discr = module_class(*args, **kwargs)
                else:
                    discr = module_class(config[idx], *args, **kwargs)
                if self.split_batch:
                    discr = mySequential(
                        BatchSplitter(no_of_heads=self.n_of_discr, head_idx=disc_idx),
                        discr)
                self.discriminators.append(discr)


    def forward_single(self, x, idx, *args, **kwargs):
        return self.discriminators[idx](x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        """Neural Network Forward Propagation with equal loss weighting."""
        # gradient weighting only in case of an attached generator:
        gen_attached = not x.is_leaf

        # split to discriminators:
        feature_dims = x.dim()
        x = x.expand([self.n_of_discr] + [-1] * feature_dims)

        # apply gradient weighting layers:
        if self.weighting_method == "soft" and gen_attached:
            # allocate tensor for discriminator predicts:
            discr_out = torch.zeros([x.shape[1], self.n_of_discr], device=x.device)  # Batch X Discr
            x = self.weight(x, self.lambda_var, discr_out) #debug detach()
        elif gen_attached:
            x = self.weight(x)

        # forward through individual discriminators:
        if self.isStudioGAN:
            result_dicts = [discr(x[idx],*args, **kwargs) for (idx, discr) in enumerate(self.discriminators)]
            x = torch.stack([result["adv_output"] for result in result_dicts], dim=-1)
        else:
            x = torch.cat([discr(x[idx],*args, **kwargs) for (idx, discr) in enumerate(self.discriminators)], dim=-1)

        # inplace update of discriminator predicts, needed for soft-weighting during Backpropagation:
        if self.weighting_method == "soft" and gen_attached:
            # adapt predicts in case of paccing (Pac-GAN):
            with torch.no_grad():
                pac_size = discr_out.shape[0] // x.shape[0]
                unpacked_x = torch.repeat_interleave(x, pac_size, dim=0)
                discr_out.add_(unpacked_x)

        # evaluation mode -> reduce predictions to single output:
        if not self.training:
            x = reduce_output(x)

        # Add additional Information to output when using StudioGAN:
        if self.isStudioGAN:
            if self.training:  # evaluation mode -> single output
                labels = torch.stack([result["label"] for result in result_dicts], dim=-1)
            else:
                labels = result_dicts[0]["label"].unsqueeze(-1)
            if x.shape != labels.shape:
                warnings.warn("Label shape does not equal output shape")
            x = {"adv_output": x, "label": labels}
        return x

class BatchSplitter(nn.Module):
    """
        This Modules implements Batch Splitting in accordance to DropoutGAN.
        Batch is equally splittet by the total number of heads.
        Depending on the head-index, a different part of the data is provided to the subsequent module.
    """

    def __init__(self, no_of_heads, head_idx):
        '''
        no_of_heads: total number of heads (discriminators)
        no_of_heads: index of this head (discriminators)
        '''
        super().__init__()
        self.no_of_heads = no_of_heads
        self.head_idx = head_idx

    def forward(self, x, labels=None, *args, **kwargs):
        if self.training:
            if x.shape[0] % self.no_of_heads != 0:
                raise ValueError("Batch size must be divisible by the number of heads")
            x = x.view(self.no_of_heads, -1, *x.shape[1:])[self.head_idx]

            if labels is not None:
                if labels.shape[0] % self.no_of_heads != 0:
                    raise ValueError("Number of labels must be divisible by the number of heads")
                labels = labels.view(self.no_of_heads, -1, *labels.shape[1:])[self.head_idx]

        if labels is None:
            out = x, *args, *kwargs
        else:
            out = x, labels, *args, *kwargs

        return out

class mySequential(nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                if len(kwargs) > 0:
                    inputs = module(*inputs, **kwargs)
                else:
                    inputs = module(*inputs)
            else:
                if len(kwargs) > 0:
                    inputs = module(inputs, **kwargs)
                else:
                    inputs = module(inputs)
        return inputs
'''
class DiscriminatorEnsemble_old(nn.Module):
    """
    Discriminator MLP-Discriminator.

    This implements a
    config-arguments:
    - discriminators: number of input features
    """

    def __init__(self, discr_class: Union[type, list[type]], config: Union[dict, list] = None, multiplier: int = 1,
                 weighting: str = 'ew', lambda_var=1, isStudioGAN=False, *args, **kwargs):
        super().__init__()
        self.isStudioGAN = isStudioGAN

        # convert discr_class to a list if single type is given:
        if not isinstance(discr_class, list):
            discr_class = [discr_class]

        # convert config to a list if only one configuration is given:
        if not isinstance(config, list):
            config = [config]

        if config is not None and len(discr_class) != len(config):
            raise ValueError("Number of given discr_classes must equal the number of configs")

        if weighting not in ["ew", "soft", "rand_uniform", "rand_normal", "rand_bernoulli"]:
            raise ValueError('Weighting must be in ["ew", "soft", "rand_uniform", "rand_normal", "rand_bernoulli"].')

        self.n_of_groups = len(discr_class)
        self.group_size = multiplier
        self.n_of_discr = len(discr_class) * multiplier
        self.lambda_var = torch.tensor(lambda_var)

        self.discriminators = nn.ModuleList()

        for idx, module_class in enumerate(discr_class):

            for _ in range(multiplier):
                if self.isStudioGAN: # -> don't use config, but pass arguments directly
                    self.discriminators.append(module_class(*args, **kwargs))
                else:
                    self.discriminators.append(module_class(config[idx], *args, **kwargs))

        self.split_and_weight = lambda x: split_and_weight.apply(x, self.n_of_discr, weighting)
        self.apply_soft_weighting = weighting == "soft"


    def forward_single(self, x, idx, *args, **kwargs):
        return self.discriminators[idx](x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        """Neural Network Forward Propagation with equal loss weighting."""
        x = self.split_and_weight(x)
        if self.apply_soft_weighting:
            # allocate tensor for discriminator predicts:
            discr_out = torch.zeros([x.shape[1], self.n_of_discr], device=x.device)  # Batch X Discr
            x = soft_weighting_autograd(x, self.lambda_var, discr_out)

        #forward through individual discriminators:
        if self.isStudioGAN:
            result_dicts = [discr(x[idx],*args, **kwargs) for (idx, discr) in enumerate(self.discriminators)]
            x = torch.stack([result["adv_output"] for result in result_dicts], dim=-1)
        else:
            x = torch.cat([discr(x[idx],*args, **kwargs) for (idx, discr) in enumerate(self.discriminators)], dim=-1)

        # take to
        if self.apply_soft_weighting:
            # manipulate discr_out Tensor that is used for soft-weighting during Backpropagation:
            # fix for pacsize:
            #ToDo fix for StudioGAN:
            pac_size = discr_out.shape[0] // x.shape[0]
            unpacced_x = torch.repeat_interleave(x, pac_size, dim=0)
            discr_out.add_(unpacced_x)

        if not self.training:  # evaluation mode -> single output
            x = reduce_output(x)

        if self.isStudioGAN:
            if self.training:  # evaluation mode -> single output
                labels = torch.stack([result["label"] for result in result_dicts], dim=-1)
            else:
                labels = result_dicts[0]["label"].unsqueeze(-1)
            if x.shape != labels.shape:
                warnings.warn("Label shape does not equal output shape")
            x = {"adv_output": x, "label": labels}
        return x
'''