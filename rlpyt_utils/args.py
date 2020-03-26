#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/24/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import argparse
import sys
import os
from pathlib import Path
from rlpyt.utils.logging.context import logger_context

import torch


def get_default_rl_parser():
    """
        Get argument parser for RL experiment.
        The current run of the experiment is stored in log_dir/name/run_id/.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    """ Structure parameters """
    parser.add_argument('-log_dir', type=str, default='data', help='Folder, where store experiments data are stored.')
    parser.add_argument('-name', type=str, default=None, help='Name of the experiment, script name is used by default.')
    parser.add_argument('-run_id', type=int, default=None, help='ID of the run. Use next ID by default in training and\
     last ID by default for evaluation.')

    """ Training parameters """
    parser.add_argument('-train_start_id', type=int, default=None, help='ID of the run from which to copy data for\
     training. Do not copy any data by default. Use -1 to use previous run data.')

    """ Evaluation parameters """
    parser.add_argument('--eval', dest='eval', action='store_true', help='Evaluate the agent.')
    parser.add_argument('--greedy_eval', dest='greedy_eval', action='store_true', help='Evaluate the agent in a greedy\
     mode.')

    return parser


def get_name(options):
    """ Return the name of the experiment from the arguments or script name without extension and absolute prefix. """
    if options.name is None:
        return sys.argv[0].split('/')[-1].split('.')[0]
    return options.name


def get_experiment_directory(options):
    """ Return directory of the experiment, i.e. log_dir/name . """
    return Path(options.log_dir).joinpath(get_name(options))


def get_last_experiment_id(options, return_minus_one_if_no_experiment=False):
    """ Return the last experiment id or None (or -1 if argument is True) if no experiment was performed. """
    exp_path = get_experiment_directory(options)
    directories = os.listdir(exp_path) if exp_path.exists() else []
    if len(directories) == 0:
        return -1 if return_minus_one_if_no_experiment else None
    ids = [int(d[4:]) for d in directories]
    return max(ids)


def is_evaluation(options):
    """ Return true if is eval or greedy_eval. """
    return options.eval or options.greedy_eval


def load_initial_model_state(options):
    """
    Return model state from previous runs.
    If evaluation, return the state for the specified run_id or the latest experiment.
    In training, return the state based on train_start_id:
        if None, return None (i.e. start training from scratch)
        if -1, use the last run_id
        otherwise, use number that is specified by train_start_id for run_id
    """
    if is_evaluation(options):
        run_id = get_last_experiment_id(options) if options.run_id is None else options.run_id
    else:
        run_id = options.train_start_id
        if run_id is not None and run_id < 0:
            run_id = get_last_experiment_id(options)
    if run_id is None:
        return None
    params_path = get_experiment_directory(options).joinpath('run_{}/params.pkl'.format(run_id))
    data = torch.load(params_path)
    return data['agent_state_dict']


def get_train_run_id(options):
    """ Return id of the run based on options. If specified, use it. Otherwise, use last_run + 1. """
    return options.run_id if options.run_id is not None else \
        get_last_experiment_id(options, return_minus_one_if_no_experiment=True) + 1


def get_default_context(options, snapshot_mode='last'):
    return logger_context(get_experiment_directory(options), get_train_run_id(options), get_name(options),
                          log_params=vars(options), snapshot_mode=snapshot_mode, use_summary_writer=True,
                          override_prefix=True)
