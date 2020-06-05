#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-5
#     Author: Kateryna Zorina and Vladimir Petrik
#
# PyTorch implementation of Dynamic Motion Primitives (DMP)
#

import torch


class DMP(torch.nn.Module):
    def __init__(self, dim=2, num_basis_functions=25,
                 alpha_x=1., dt=0.01, max_time=1., tau=1.,
                 alpha_y=25., beta_y=6.25, weight_scale=1e3,
                 dtype=torch.float32):
        """
            Module for multidimensional DMP with shared time value.

            The time is controlled by canonical system defined by alpha_x, and dt parameters.
            The basis functions are located uniformly in the time span defined by max_time.
        """
        super().__init__()

        self.dim = dim

        """ Time values """
        self.x = torch.tensor(1., dtype=dtype)  # Internal time counter
        self.tau = torch.tensor(tau, dtype=dtype)
        self.alpha_x = alpha_x
        self.dt = dt
        self.max_time = max_time

        """ Basis functions PSI """
        self.c = torch.exp(-self.alpha_x * torch.linspace(0., max_time, num_basis_functions, dtype=dtype))
        self.h = torch.ones(num_basis_functions, dtype=dtype) * num_basis_functions / self.c

        """ Goal and weights parameters """
        self.g = torch.nn.Parameter(torch.ones(dim, dtype=dtype))
        self.weights = torch.nn.Parameter(torch.ones((dim, num_basis_functions), dtype=dtype) / num_basis_functions)
        self.weight_scale = torch.tensor(weight_scale, dtype=dtype)

        """ PD controller setup for the trajectory following. """
        self.alpha_y = torch.tensor(alpha_y, dtype=dtype)
        self.beta_y = torch.tensor(beta_y, dtype=dtype)

    def psi(self):
        """ Get all basis functions for current time x. """
        return torch.exp(-self.h * (self.x - self.c) ** 2)

    def reset(self):
        """ Reset internal time counter to 1. """
        self.x.data.fill_(1.)

    def decay_time(self):
        """ Update time by one step. """
        self.x = self.x - self.alpha_x * self.x * self.tau * self.dt

    def force(self, y0):
        """ Return force for current time and given y0. """
        psi = self.psi()
        return torch.sum(self.weight_scale * self.weights * psi, dim=-1) * self.x * (self.g - y0) / torch.sum(psi)

    def forward(self, state, with_time_decay=True):
        """
            Perform one step from the current state and internal time variable.
            State must consists of: [y0, y_t, dy_t], same values are returned for the time t+1.
         """
        y0, y, dy = state[:self.dim], state[self.dim:2 * self.dim], state[2 * self.dim:3 * self.dim]
        if with_time_decay:
            self.decay_time()
        ddy = self.alpha_y * (self.beta_y * (self.g - y) - dy) + self.force(y0)
        ndy = dy + self.tau * ddy * self.dt
        ny = y + self.tau * ndy * self.dt
        return torch.cat([y0, ny, ndy])

    def get_rollout_buffer(self, time=None, dtype=torch.float32):
        """ Create a buffer for rollout based on dt and max_time. """
        time = self.max_time if time is None else time
        return torch.zeros((int(time // self.dt) + 1, self.dim * 3), dtype=dtype)

    def rollout(self, y0, time=None, buffer=None):
        """ Call forward function to generate trajectory with length equal to max_time given by constructor. """
        self.reset()
        if buffer is None:
            buffer = self.get_rollout_buffer(time, dtype=y0.dtype)
        buffer[0, :self.dim] = y0
        buffer[0, self.dim:2 * self.dim] = y0
        buffer[0, 2 * self.dim:3 * self.dim].fill_(0)
        for i in range(1, buffer.shape[0]):
            buffer[i] = self.forward(buffer[i - 1])
        return buffer

    def fix_goal(self, g):
        """ Fig goal value by turning off gradient optimization and by manually setting its value. """
        self.g.data = g
        self.g.requires_grad = False
