#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/23/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import numpy as np


class DiscreteAction:
    """ Class that represent key-value discrete action. Key is anything human readable and is used to apply action. """

    def __init__(self, key, val=None) -> None:
        self.key = key
        self.val = val


def append_discretized_nd_action(actions, name='spd', n=3, max_action=1., spd_scales=None):
    """ Append actions from n-dimensional space. Only orthogonal actions are used. """
    if spd_scales is None:
        spd_scales = [1.]

    for i in range(n):
        for sig in [-1, 1]:
            for scale in spd_scales:
                v = np.zeros(n)
                v[i] = scale * sig * max_action
                actions.append(DiscreteAction(name, v))
