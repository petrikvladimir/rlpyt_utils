#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/23/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>


import os
import numpy as np
import torch
from rlpyt.utils.logging import logger


def exponential_reward(vec, b=1, scale=1, axis=-1):
    """ Compute exponential reward in a form: scale * exp(-0.5 * b * || v||^2) """
    return scale * np.exp(-0.5 * b * np.linalg.norm(vec, axis=axis) ** 2)


def load_saved_params(log_dir, run_id, exp_date=None):
    """
    Load parameters from the experiment either specified by the exp date or the latest.
    The
    """
    if exp_date is None:
        exps = os.listdir(log_dir)
        exps.sort(reverse=True)
        exp_date = exps[0]
    logger.log('Using the experiment with timestamp: {}'.format(exp_date))
    log_dir_div = '' if log_dir[-1] == '/' else '/'
    params_path = '{}{}{}/run_{}/params.pkl'.format(log_dir, log_dir_div, exp_date, run_id)
    return torch.load(params_path)
