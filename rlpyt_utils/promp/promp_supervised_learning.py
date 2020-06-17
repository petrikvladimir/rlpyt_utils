#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-16
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from rlpyt_utils.promp.promp import ProMP
import torch
import matplotlib.pyplot as plt

# Create multiple demonstrations
ref_time = torch.linspace(0., 10.)
ref_y = torch.stack([torch.sin(ref_time), torch.cos(0.3 * ref_time)], dim=-1)  # [T x D]
ref_y = ref_y.unsqueeze(-1).repeat(1, 1, 5)  # [T x D x B]
ref_y += torch.randn(2, 5) * 0.1
ref_y += torch.randn_like(ref_y) * 0.05

promp = ProMP(n_dof=2, num_basis_functions=100, t_start=ref_time[0], t_stop=ref_time[-1],
              init_scale_cov_w=1e-2, init_scale_mu_w=1e-2, cov_w_is_diagonal=False, position_only=True)

promp.set_params_from_reference_trajectories(ref_y, ref_time, lambda_eps=1e-1)

y = promp.mu_and_cov_y(ref_time)[0].detach().numpy()
y_samples = promp.sample_trajectories(num_samples=100, t=ref_time)

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(y_samples[:, :, 0], y_samples[:, :, 1], '-', alpha=0.1, color='tab:blue')
ax.plot(y[:, 0], y[:, 1], '-', label='Pos ProMP', color='tab:green')
ax.plot(ref_y[:, 0], ref_y[:, 1], '--', label='Pos Reference', color='tab:orange')
ax.legend()
ax.axis('equal')

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(ref_time, y, '-', label='ProMP')

for i in range(ref_y.shape[-1]):
    ax.set_prop_cycle(None)
    ax.plot(ref_time, ref_y[:, :, i], '--', label='Ref')
ax.legend()

plt.show()
