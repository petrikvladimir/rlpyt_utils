#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 4/6/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from rlpyt.runners.minibatch_rl import MinibatchRl


class MinibatchRlWithLog(MinibatchRl):
    """
    Extends original minibatchRL with a functor calling at log diagnostics.
    """

    def __init__(self, log_traj_window=100, log_diagnostics_fun=None, store_diagnostics_fun=None, **kwargs):
        """
        :param log_traj_window: passed to original MinibatchRL
        :param log_diagnostics_fun: callable in form fun(itr, algo, agent, sampler) called at each store diagnostics
        :param store_diagnostics_fun: callable in form fun(itr, algo, agent, sampler) called at each store diagnostics
        """
        super().__init__(log_traj_window, **kwargs)
        self.store_diagnostics_fun = store_diagnostics_fun
        self.log_diagnostics_fun = log_diagnostics_fun

    def store_diagnostics(self, itr, traj_infos, opt_info):
        super().store_diagnostics(itr, traj_infos, opt_info)
        if self.store_diagnostics_fun is not None:
            self.store_diagnostics_fun(itr, self.algo, self.agent, self.sampler)

    def log_diagnostics(self, itr):
        super().log_diagnostics(itr)
        if self.log_diagnostics_fun is not None:
            self.log_diagnostics_fun(itr, self.algo, self.agent, self.sampler)
