#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-9
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
from typing import Optional

import torch
import torch.distributions as tp


class ProMP(torch.nn.Module):

    def __init__(self, n_dof, num_basis_functions=25, t_start=0., t_stop=1.,
                 init_scale_cov_w=1e-2, init_scale_mu_w=1.,
                 cov_eps=1e-7, sigma_y=1e-7, cov_w_is_diagonal=False,
                 position_only=False) -> None:
        super().__init__()

        self.position_only = position_only

        self.N = num_basis_functions  # N - number of basis functions
        self.D = n_dof  # D - number of system DOF
        self.mD = self.D if self.position_only else 2 * self.D  # 2D if pos+vel, 1D if pos only

        heq = 1 / self.N
        self.c = torch.linspace(-2 * heq, 1 + 2 * heq, self.N)
        self.h = heq * torch.ones(self.N)

        self.cov_eps = cov_eps
        self.mu_w_params = torch.nn.Parameter(init_scale_mu_w * torch.rand(self.N * self.D))
        self.cov_w_params = torch.nn.Parameter(init_scale_cov_w * (
            torch.rand((self.N * self.D) if cov_w_is_diagonal else (self.N * self.D, self.N * self.D))
        ))
        self.sigma_y = sigma_y * torch.eye(self.mD).unsqueeze(0)  # 1 x mD x mD
        self.conditioning = []
        self.t_start = t_start
        self.t_stop = t_stop

    @property
    def is_conditioned(self):
        return len(self.conditioning) != 0

    @property
    def mu_and_cov_w(self):
        mu = self.mu_w_params
        cov_w_params = torch.diag(self.cov_w_params) if len(self.cov_w_params.shape) == 1 else self.cov_w_params
        cov = cov_w_params.matmul(cov_w_params.transpose(0, 1)) + self.cov_eps * torch.eye(self.N * self.D)
        for cond in self.conditioning:
            mu, cov = self._condition_mu_and_cov(mu, cov, **cond)
        return mu, cov

    def phase(self, t, t_start=None, t_end=None):
        """ Compute linear phase s.t. z=0 for t=t_start and z = 1 for t=t_end. Returns z and dz."""
        if t_start is None:
            t_start = self.t_start
        if t_end is None:
            t_end = self.t_stop
        assert t_end > t_start
        assert torch.all(t >= t_start)
        assert torch.all(t <= t_end)
        return (t - t_start) / (t_end - t_start), torch.ones_like(t) / (t_end - t_start)

    def get_phi_tensor(self, t: torch.Tensor):
        """ For given time stamps, compute phi tensor TxNxm, where N is number of basis, and last dimension represents
            position and/or velocity.
        """
        p, dp = self._psi(*self.phase(t), self.c, self.h)
        if self.position_only:
            return p.unsqueeze(-1)
        phi = torch.stack((p, dp), dim=-1)
        return phi

    def get_psi_matrix(self, t: torch.Tensor):
        """ Get block diagonal Psi matrix used in the paper. Return T x DN x mD matrix for all T. """
        phi = self.get_phi_tensor(t)
        return self._block_diag(*[phi for _ in range(self.D)])

    def mu_and_cov_y(self, t: Optional[torch.Tensor] = None, phi: Optional[torch.Tensor] = None):
        """
        Compute mean and covariance either from time steps or from precomputed phi.
        Only one of them needs to be specified.
        :param t: time steps [T]
        :param phi: phi tensor with shapes [T, N, m] see get_phi_tensor()
        :returns [T x mD x mD] covariance matrices for y; first m correspond to 1. DOF, next m to 2. DOF, etc.
        """
        assert t is None or phi is None
        assert not (t is None and phi is None)
        phi = phi if phi is not None else self.get_phi_tensor(t)
        phi = phi.unsqueeze(1).unsqueeze(1)  # [T x 1 x 1 x N x 2]
        phi_t = phi.transpose(-2, -1)  # [T x 1 x 1 x 2 x N]
        mu_w, cov_w = self.mu_and_cov_w

        mu_w_blk = mu_w.reshape((1, 1, self.D, self.N, 1))  # [1 X 1 x D x N x 1]
        mu_y_blk = torch.matmul(phi.transpose(-2, -1), mu_w_blk)  # [T x 1 x D x 2 x 1]
        mu_y = mu_y_blk.reshape(-1, self.mD)

        cov_w_blk = cov_w.reshape(1, self.D, self.N, self.D, self.N).transpose(-3, -2)  # [1 x D x D x N x N]
        cov_y_blk = torch.matmul(phi_t, torch.matmul(cov_w_blk, phi))  # T x D x D x m x m
        cov_y = cov_y_blk.transpose(2, 3).reshape(-1, self.mD, self.mD) + self.sigma_y  # T x mD x mD

        return mu_y, cov_y

    def w_dist(self):
        """ Returns distribution for weights. Multivariate gaussian N(mu_w, cov_w)."""
        return tp.MultivariateNormal(*self.mu_and_cov_w)

    def y_dist(self, t: Optional[torch.Tensor] = None, phi: Optional[torch.Tensor] = None):
        """ Returns distribution for y. Batch of multivariate gaussians N(mu_y, cov_y) for each time or phi."""
        return tp.MultivariateNormal(*self.mu_and_cov_y(t, phi))

    def condition(self, t: float, y: torch.Tensor = None, cov: torch.Tensor = None):
        """
            Condition ProMP to reach given state y with precision given by cov.
            T must be scalar tensor. y must by vector with 2D elements, and cov is 2Dx2D precision matrix.
        """
        self.conditioning.append(dict(t=t, y=y, cov=cov))

    def _condition_mu_and_cov(self, mu_w: torch.Tensor, cov_w: torch.Tensor, t: float,
                              y: torch.Tensor = None, cov: torch.Tensor = None):
        """
            Condition ProMP to reach given state y with precision given by cov.
            T must be scalar tensor. y must by vector with 2D elements, and cov is 2Dx2D precision matrix.
        """
        psi = self.get_psi_matrix(t=torch.tensor(t))
        psi_t = psi.transpose(-2, -1)
        mtmp = cov_w.mm(psi).mm(torch.inverse(cov + psi_t.mm(cov_w).mm(psi)))
        new_mu_w = mu_w + mtmp.mv(y - psi_t.mv(mu_w))
        new_cov_w = cov_w - mtmp.mm(psi_t.mm(cov_w))
        return new_mu_w, new_cov_w

    def sample_trajectories(self, t, num_samples=10):
        """
            Sample [num_samples] trajectories from a weights distribution.
            Method first sample weights from gaussian distribution and then transform them to DOF space using basis.
            :returns [T x num_samples x 2D ] tensor.
        """
        w = self.w_dist().sample(sample_shape=(num_samples,)).reshape(1, -1, self.D, self.N, 1)  # [1 x B x D x N x 1]
        phi_t = self.get_phi_tensor(t).unsqueeze(1).unsqueeze(1).transpose(-2, -1)
        y = phi_t.matmul(w).reshape(-1, num_samples, self.mD)
        return y

    @staticmethod
    def _psi(z: torch.Tensor, zdot: torch.Tensor, c: torch.Tensor, h: torch.Tensor):
        """
            Compute psi and dpsi function for given phase variable z [B1 x .. x Bn] and given parameters
            h [C1 x ... x Cn] and c [C1 x ... x Cn].
            Returns [B1 x ... x Bn x C1 x ... x Cn]. I.e. for all times, compute the psi.
        """
        b_ndim = len(h.shape)
        k = (z[(...,) + (None,) * b_ndim] - c)
        b = torch.exp(-0.5 * (k ** 2) / h)
        dot_b = -k / h * zdot[(...,) + (None,) * b_ndim] * b
        return b, dot_b

    @staticmethod
    def _block_diag(*arrs):
        """ https://github.com/pytorch/pytorch/issues/31932 """
        shapes = torch.tensor([a.shape for a in arrs])
        out_shape = shapes[0, :-2].tolist() + torch.sum(shapes, dim=0)[-2:].tolist()
        out = torch.zeros(out_shape, dtype=arrs[0].dtype, device=arrs[0].device)
        r, c = 0, 0
        for i, (rr, cc) in enumerate(shapes[:, -2:]):
            out[..., r:r + rr, c:c + cc] = arrs[i]
            r += rr
            c += cc
        return out
