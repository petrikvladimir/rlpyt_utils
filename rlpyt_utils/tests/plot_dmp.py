#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-5
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

# !/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from rlpyt_utils.dmp import DMP
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

dmp = DMP(dim=2, dtype=torch.float32, dt=0.05, max_time=5., num_basis_functions=250, weight_scale=1e3)

time = np.arange(0., dmp.max_time, dmp.dt).astype(np.float32)
ref_trajectory = np.stack([np.sin(time), np.cos(time)]).T
ref_trajectory[-10:, :] = ref_trajectory[-10, :]

y0 = torch.tensor(ref_trajectory[0])
dmp.fix_goal(torch.tensor(ref_trajectory[-1]))
# dmp.g.data = torch.tensor(ref_trajectory[-1])

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(dmp.parameters(), lr=5e-2, amsgrad=False)

bar = trange(200)
for epoch in bar:
    dmp.reset()
    optimizer.zero_grad()
    trajectory = dmp.rollout(y0)
    loss = loss_fn(trajectory[:, 2:4], torch.from_numpy(ref_trajectory))
    bar.set_postfix_str('loss: {}'.format(loss))
    loss.backward()
    optimizer.step()
    if loss.detach() < 1e-3:
        break

trajectory = dmp.rollout(y0).detach()
fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(trajectory[:, 2], trajectory[:, 3], label='DMP trajectory', color='tab:green')
ax.plot(trajectory[0, 2], trajectory[0, 3], 'o', label='DMP  start', color='tab:green')
ax.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], ':', label='Ref trajectory', color='tab:blue')
ax.plot(ref_trajectory[0, 0], ref_trajectory[0, 1], 'o', label='Ref start', color='tab:blue')
g = dmp.g.detach()
ax.plot(g[0], g[1], 'o', label='DMP goal', color='tab:red')
ax.axis('equal')
ax.legend()

dmp.tau = 0.5
trajectory2 = dmp.rollout(y0, time=15).detach()
fig, axes = plt.subplots(2, 1, squeeze=False, sharex=True, sharey=True)  # type: plt.Figure, Array[plt.Axes]
ax = axes[0, 0]
t = np.linspace(0, dmp.max_time, ref_trajectory.shape[0])
ax.plot(t, trajectory[:, 2], label='DMP')
ax.plot(np.linspace(0, 15, trajectory2.shape[0]), trajectory2[:, 2], label='DMP tau=0.5')
ax.plot(t, ref_trajectory[:, 0], label='Ref')
ax.legend()
ax.set_xlabel('time [s]')

ax = axes[1, 0]
ax.plot(t, trajectory[:, 3], label='DMP')
ax.plot(np.linspace(0, 15, trajectory2.shape[0]), trajectory2[:, 3], label='DMP tau=0.5')
ax.plot(t, ref_trajectory[:, 1], label='Ref')
ax.legend()
ax.set_xlabel('time [s]')

plt.show()
