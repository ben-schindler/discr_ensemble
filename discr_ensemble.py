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
from typing import Union
from easydict import EasyDict as edict

from ensemble_utils import reduce_output, split_and_weight, soft_weighting, soft_weighting_autograd

'''
from functorch import combine_state_for_ensemble
from functorch import vmap

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
                 weighting: str = 'ew', lambda_var=1, *args, **kwargs):
        super().__init__()

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
                self.discriminators.append(module_class(config[idx], *args, **kwargs))

            #code for vectorized group
            #if multiplier>1:
            #    group = DiscriminatorGroup(discr_class=module_class, multiplier=multiplier,  config=config[idx], *args, **kwargs)
            #    self.discriminators.append(group)
            #else:
            #    self.discriminators.append(module_class(config[idx], *args, **kwargs))

        self.split_and_weight = lambda x: split_and_weight.apply(x, self.n_of_discr, weighting)
        self.apply_soft_weighting = weighting == "soft"


    def forward_single(self, x, idx):
        return self.discriminators[idx](x)

    def forward(self, x):
        """Neural Network Forward Propagation with equal loss weighting."""
        x = self.split_and_weight(x)
        if self.apply_soft_weighting:
            discr_out = torch.zeros([x.shape[1], self.n_of_discr])  # Batch X Discr
            x = soft_weighting_autograd(x, self.lambda_var, discr_out)
        x = torch.concat([discr(x[idx]) for (idx, discr) in enumerate(self.discriminators)], dim=-1)

        if self.apply_soft_weighting:
            # manipulate discr_out Tensor that is used for soft-weighting during Backpropagation:
            # fix for pacsize:
            pac_size = discr_out.shape[0] // x.shape[0]
            unpacced_x = torch.repeat_interleave(x, pac_size, dim=0)
            discr_out.add_(unpacced_x)

        if not self.training:  # evaluation mode -> single output
            x = reduce_output(x)
        return x
