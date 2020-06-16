#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from rlpyt_utils.promp.promp import ProMP
import torch
import matplotlib.pyplot as plt

promp = ProMP(1, init_scale_cov_w=1e-1)

t = torch.linspace(promp.t_start, promp.t_stop)

fig, axes = plt.subplots(1, 3, squeeze=True, figsize=(6.4 * 2, 4.8))

y_sample = promp.sample_trajectories(t, num_samples=100).detach().cpu().numpy()
axes[0].plot(t, y_sample[:, :, 0, 0], '-', label='Position', color='tab:blue', alpha=0.5)
axes[0].plot(t, y_sample[:, :, 0, 1], ':', label='Velocity', color='tab:orange', alpha=0.5)
axes[0].set_title('Random weights')

promp.condition(0., torch.zeros(2), torch.diag(torch.tensor((1e-3, 1000.))))
y_sample = promp.sample_trajectories(t, num_samples=100).detach().cpu().numpy()
axes[1].plot(t, y_sample[:, :, 0, 0], '-', label='Position', color='tab:blue', alpha=0.5)
axes[1].plot(t, y_sample[:, :, 0, 1], ':', label='Velocity', color='tab:orange', alpha=0.5)
axes[1].set_title('Conditioning for pos')

promp.condition(0., torch.zeros(2), 1e-3 * torch.eye(2))
y_sample = promp.sample_trajectories(t, num_samples=100).detach().cpu().numpy()
axes[2].plot(t, y_sample[:, :, 0, 0], '-', label='Position', color='tab:blue', alpha=0.5)
axes[2].plot(t, y_sample[:, :, 0, 1], ':', label='Velocity', color='tab:orange', alpha=0.5)
axes[2].set_title('Conditioning for pos and vel')

for ax in axes:
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('y [-]')

plt.show()
