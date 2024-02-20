"""
IGN_Discriminators.

Discriminator Models for Impedance Spectra.
date: 31.05.2022
author: Benjamin Schindler
"""
import torch
import torch.nn as nn
import torchinfo
from functorch import combine_state_for_ensemble
from functorch import vmap

import json
import copy
from typing import Union
from easydict import EasyDict as edict

from ensemble_utils import reduce_output




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


class DiscriminatorEnsemble(nn.Module):
    """
    Discriminator MLP-Discriminator.

    This implements a
    config-arguments:
    - discriminators: number of input features
    """

    def __init__(self, discr_class: Union[type, list[type]], config: Union[dict, list] = None, multiplier: int = 1,
                 weighting: str = 'ew', *args, **kwargs):
        super().__init__()

        # convert discr_class to a list if single type is given:
        if not isinstance(discr_class, list):
            discr_class = [discr_class]

        # convert config to a list if only one configuration is given:
        if not isinstance(config, list):
            config = [config]

        if config is not None and len(discr_class) != len(config):
            raise ValueError("Number of given discr_classes must equal the number of configs")

        if weighting not in ["ew"]:
            raise ValueError("Weighting must be ew.")

        self.n_of_groups = len(discr_class)
        self.group_size = multiplier
        self.n_of_discr = len(discr_class) * multiplier

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




    def forward(self, x):
        """Neural Network Forward Propagation with equal loss weighting."""
        x = torch.concat([discr(x) for discr in self.discriminators], dim=-1)
        if not self.training: # evaluation mode -> single output
            x = reduce_output(x)
        return x

    def forward_single(self, x, idx):
        return self.discriminators[idx](x)

    def forward_group(self, x, idx):
        group_indices = range(idx*self.group_size, (idx+1)*self.group_size)
        x = torch.concat([self.discriminators[i](x) for i in group_indices], dim=-1)
        if not self.training: # evaluation mode -> single output
            x = reduce_output(x)
        return x

    def _forward_aggregated_ew(self, x):
        """Neural Network Forward Propagation with equal loss weighting."""
        x = torch.stack([self.discriminators[key](x) for key in self.discriminators.keys()])
        x = reduce_output(x)
        return x

    def _forward_aggregated_rf(self, x):
        """Neural Network Forward Propagation with equal loss weighting."""
        x = torch.stack([self.discriminators[key](x) for key in self.discriminators.keys()])
        x = reduce_output(x)
        return x