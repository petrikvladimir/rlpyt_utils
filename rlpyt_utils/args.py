#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/24/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import argparse
import sys
import os
from pathlib import Path

import psutil
from rlpyt.algos.pg.ppo import PPO
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.logging import logger

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

    """ Affinity parameters """
    parser.add_argument('-cuda_id', type=int, default=None, help='ID of cuda device used for training. ')

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


def load_initial_model_state(options, structure=None):
    """
    Return model state from previous runs.
    If evaluation, return the state for the specified run_id or the latest experiment.
    Structure specifies the hierarchy in data ['agent_state_dict'] by default, i.e. structure for PPO.
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
    if structure is None:
        return data['agent_state_dict']
    d = data
    for s in structure:
        d = d[s]
    return d


def get_train_run_id(options):
    """ Return id of the run based on options. If specified, use it. Otherwise, use last_run + 1. """
    return options.run_id if options.run_id is not None else \
        get_last_experiment_id(options, return_minus_one_if_no_experiment=True) + 1


def get_default_context(options, snapshot_mode='last', snapshot_gap=500):
    if snapshot_mode == 'gap':
        logger.set_snapshot_gap(snapshot_gap)
    return logger_context(get_experiment_directory(options), get_train_run_id(options), get_name(options),
                          log_params=vars(options), snapshot_mode=snapshot_mode, use_summary_writer=True,
                          override_prefix=True)


def add_default_ppo_args(parser, lr=3e-4, epochs=25, ratio_clip=0.2, gae=0.95, discount=0.99, entropy=1e-8,
                         clip_grad_norm=1e8):
    """ Add default PPO arguments s.t. get_ppo_from_options can be used. """
    parser.add_argument('-discount', type=float, default=discount)
    parser.add_argument('-lr', type=float, default=lr)
    parser.add_argument('-entropy', type=float, default=entropy)
    parser.add_argument('-epochs', type=int, default=epochs)
    parser.add_argument('-ratio_clip', type=float, default=ratio_clip)
    parser.add_argument('-gae', type=float, default=gae)
    parser.add_argument('-clip_grad_norm', type=float, default=clip_grad_norm)
    parser.add_argument('--linear_lr', dest='linear_lr', action='store_true')


def get_ppo_from_options(options, **kwargs):
    """ Get ppo algorithm from options. """
    algo = PPO(entropy_loss_coeff=options.entropy, learning_rate=options.lr, discount=options.discount,
               gae_lambda=options.gae, ratio_clip=options.ratio_clip, epochs=options.epochs,
               clip_grad_norm=options.clip_grad_norm, linear_lr_schedule=options.linear_lr, **kwargs)
    return algo


def get_affinity(options):
    """ Get affinity for all available processors together with cuda id specified by options. """
    p = psutil.Process()
    affinity = dict(cuda_idx=options.cuda_id, workers_cpus=p.cpu_affinity())
    return affinity
