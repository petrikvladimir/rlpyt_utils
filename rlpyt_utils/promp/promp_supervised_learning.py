#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-16
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from rlpyt_utils.promp.promp import ProMP
import torch
import matplotlib.pyplot as plt
from nptyping import Array
from tqdm import trange

ref_time = torch.linspace(0., 10.)
ref_y = torch.stack([
    torch.sin(ref_time),
    torch.cos(0.3 * ref_time),
], dim=-1)

promp = ProMP(n_dof=2, num_basis_functions=100, t_start=ref_time[0], t_stop=ref_time[-1],
              init_scale_cov_w=1e-2, init_scale_mu_w=1e-2, cov_w_is_diagonal=True, position_only=True)
# promp.condition(float(ref_time[0]), ref_y[0], 1e-4 * torch.eye(4))

# promp.cov_w_params.requires_grad = False
# promp.cov_w_params.data = 1e-3 * torch.ones_like(promp.cov_w_params.data)

optimizer = torch.optim.Adam(promp.parameters(), lr=1e-3 if promp.is_conditioned else 1e-3)

phi = promp.get_phi_tensor(ref_time)
bar = trange(1000)
for epoch in bar:
    optimizer.zero_grad()
    dist = promp.y_dist(phi=phi)
    neg_log_prob = -dist.log_prob(ref_y).sum()
    bar.set_postfix_str('Neg log prob: {}'.format(neg_log_prob))
    neg_log_prob.backward()
    optimizer.step()

promp.condition(0., ref_y[0] + torch.tensor([1., 1.]), 1e-4 * torch.eye(2))

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
y = promp.mu_and_cov_y(ref_time)[0].detach().numpy()

y_samples = promp.sample_trajectories(ref_time, num_samples=100)
ax.plot(y_samples[:, :, 0], y_samples[:, :, 1], '-', alpha=0.15, color='tab:blue')

ax.plot(y[:, 0], y[:, 1], '-', label='Pos ProMP', color='tab:green')
ax.plot(ref_y[:, 0], ref_y[:, 1], '--', label='Pos Reference', color='tab:orange')

ax.legend()
ax.axis('equal')

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(ref_time, y, '-', label='ProMP')
ax.set_prop_cycle(None)
ax.plot(ref_time, ref_y, '--', label='Ref')
ax.legend()

plt.show()
