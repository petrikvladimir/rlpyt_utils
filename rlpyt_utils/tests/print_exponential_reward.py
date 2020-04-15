#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 4/15/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>


from rlpyt_utils.utils import exponential_reward
import numpy as np

dists = [0.01, 0.05, 0.1, 0.5, 1., 5., 10.]
bs = [0.01, 0.1, 10, 100, 1000, 10000]

dd, bb = np.meshgrid(dists, bs)
v = np.empty_like(dd)
for i in range(dd.shape[0]):
    for j in range(dd.shape[1]):
        v[i, j] = exponential_reward([dd[i, j]], b=bb[i, j])

# print(dd)
# print(bb)
# print(v)


for i in range(-2, dd.shape[0]):
    for j in range(-1, dd.shape[1]):
        if i == -2 and j < 0:
            print('exp', end=' | ' if j != dd.shape[1] - 1 else '')
        if i == -2 and j >= 0:
            print('d={}'.format(dd[0, j]), end=' | ' if j != dd.shape[1] - 1 else '')
        if i == -1:
            print('---', end=' | ' if j != dd.shape[1] - 1 else '')

        if j < 0 and i >= 0:
            print('**b={:.0e}**'.format(bb[i, 0]), end=' | ' if j != dd.shape[1] - 1 else '')
        elif j >= 0 and i >= 0:
            # print('{:.1e}'.format(v[i, j]), end=' | ' if j != dd.shape[1] - 1 else '')
            print('{}'.format(round(v[i, j], 2)), end=' | ' if j != dd.shape[1] - 1 else '')
    print('')
