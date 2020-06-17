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
        self.M = 1 if self.position_only else 2
        self.mD = self.M * self.D

        heq = 1 / self.N
        self.c = torch.linspace(-2 * heq, 1 + 2 * heq, self.N)
        self.h = heq * torch.ones(self.N)

        self.cov_eps = cov_eps
        self.mu_w_params = torch.nn.Parameter(init_scale_mu_w * torch.randn(self.N * self.D))
        self.cov_w_params = torch.nn.Parameter(init_scale_cov_w * (
            torch.randn((self.N * self.D) if cov_w_is_diagonal else (self.N * self.D, self.N * self.D))
        ))
        self.sigma_y = sigma_y * torch.eye(self.mD).unsqueeze(0)  # 1 x mD x mD
        self.conditioning = []
        self.t_start = t_start
        self.t_stop = t_stop

    @property
    def is_conditioned(self):
        """ Return true if promp is conditioned, i.e. conditioning list is not empty. """
        return len(self.conditioning) != 0

    @property
    def is_cov_diagonal(self):
        return len(self.cov_w_params.shape) == 1

    @property
    def mu_and_cov_w(self):
        """
        Computes, possibly conditioned, mean and covariance for the weights.
        :returns mu [ND], cov [ND x ND]
        """
        mu = self.mu_w_params
        cov_w_params = torch.diag(self.cov_w_params) if self.is_cov_diagonal else self.cov_w_params
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
        """
        For given time instances [T] computes phi matrices.
        :returns phi [T x M x N]
        """
        p, dp = self._psi(*self.phase(t), self.c, self.h)
        if self.position_only:
            return p.unsqueeze(-2)
        phi = torch.stack((p, dp), dim=-2)
        return phi

    def get_psi_matrix(self, t: Optional[torch.Tensor] = None, phi: Optional[torch.Tensor] = None):
        """
        Get block diagonal Psi matrix used in the paper.
        Return T x DM x DN matrix for all T.
        """
        phi = self._get_unsqueezed_phi(t, phi, unsqueeze=False)
        return self._block_diag(*[phi for _ in range(self.D)])

    def _get_unsqueezed_phi(self, t: Optional[torch.Tensor] = None, phi: Optional[torch.Tensor] = None, unsqueeze=True):
        """
        Internal function to compute unsqueezed phi either from time or from precomputed phi.
        :param t: [T]
        :param phi: [T x M x N]
        :return: [T x 1 x 1 x M x N] if unsqueeze True else [T x M x N]
        """
        assert t is None or phi is None
        assert not (t is None and phi is None)
        phi = phi if phi is not None else self.get_phi_tensor(t)
        if unsqueeze:
            return phi.unsqueeze(1).unsqueeze(1)
        else:
            return phi

    def y_from_weights(self, w: torch.Tensor, t: Optional[torch.Tensor] = None, phi: Optional[torch.Tensor] = None):
        """
        Computes y from given weights and either time or phi. Batched computation of Phi * w
        :param w: [DN]
        :returns y [T x MD]
        """
        phi = self._get_unsqueezed_phi(t, phi)  # [T x 1 x 1 x M x N]
        w_blk = w.reshape((1, 1, self.D, self.N, 1))  # [1 X 1 x D x N x 1]
        y_blk = phi.matmul(w_blk)  # [T x 1 x D x M x 1]
        return y_blk.reshape(-1, self.mD)

    def mu_and_cov_y(self, t: Optional[torch.Tensor] = None, phi: Optional[torch.Tensor] = None):
        """
        Compute mean and covariance either from time steps or from precomputed phi.
        Only one of them needs to be specified.
        :param t: time steps [T]
        :param phi: phi tensor with shapes [T x M x N]
        :returns mu [T x DM], cov [T x DM x DM]
        """
        phi = self._get_unsqueezed_phi(t, phi)  # [T x 1 x 1 x M x N]
        phi_t = phi.transpose(-2, -1)  # [T x 1 x 1 x N x M]
        mu_w, cov_w = self.mu_and_cov_w

        mu_w_blk = mu_w.reshape((1, 1, self.D, self.N, 1))  # [1 X 1 x D x N x 1]
        mu_y_blk = phi.matmul(mu_w_blk)  # [T x 1 x D x M x 1]
        mu_y = mu_y_blk.reshape(-1, self.mD)

        cov_w_blk = cov_w.reshape(1, self.D, self.N, self.D, self.N).transpose(-3, -2)  # [1 x D x D x N x N]
        cov_y_blk = phi.matmul(cov_w_blk).matmul(phi_t)  # T x D x D x m x m
        cov_y = cov_y_blk.transpose(2, 3).reshape(-1, self.mD, self.mD) + self.sigma_y  # T x DM x DM

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
        L = cov_w.mm(psi_t).mm(torch.inverse(cov + psi.mm(cov_w).mm(psi_t)))
        new_mu_w = mu_w + L.mv(y - psi.mv(mu_w))
        new_cov_w = cov_w - L.mm(psi.mm(cov_w))
        return new_mu_w, new_cov_w

    def sample_trajectories(self, num_samples=10, t: Optional[torch.Tensor] = None, phi: Optional[torch.Tensor] = None):
        """
            Sample [num_samples] trajectories from a weights distribution.
            Method first sample weights from gaussian distribution and then transform them to DOF space using basis.
            :returns [T x num_samples x DM ] tensor.
        """
        w = self.w_dist().sample(sample_shape=(num_samples,)).reshape(1, -1, self.D, self.N, 1)  # [1 x B x D x N x 1]
        phi = self._get_unsqueezed_phi(t, phi)
        y = phi.matmul(w).reshape(-1, num_samples, self.mD)
        return y

    # def weights_from_trajectory(self, ref_y: torch.Tensor, t: Optional[torch.Tensor] = None,
    #                             phi: Optional[torch.Tensor] = None):
    #     """
    #     Computes weights from given reference trajectory. Solve least square problem A w = b.
    #     :param ref_y [T x DM x B] or [T x DM]
    #     :param t [T]
    #     :param phi [T x M x N]
    #     :return: weights [DN x B], i.e. weights for all reference trajectories.
    #     """
    #     psi = self.get_psi_matrix(t, phi)  # [T x DM x DN]
    #     A = psi.reshape(-1, self.D * self.N)  # [TDM x DN]
    #     b = ref_y.reshape(A.shape[0], -1)  # [TDM x B]
    #     X, _ = torch.lstsq(b, A)
    #     return X[:A.shape[1]]

    def weights_from_trajectory(self, ref_y: torch.Tensor, t: Optional[torch.Tensor] = None,
                                      phi: Optional[torch.Tensor] = None, lambda_eps=1e-3):
        """
        Computes weights from given reference trajectory. Solve least square problem A w = b.
        :param lambda_eps: regularization
        :param ref_y [T x DM x B] or [T x DM]
        :param t [T]
        :param phi [T x M x N]
        :return: weights [DN x B], i.e. weights for all reference trajectories.
        """
        psi = self.get_psi_matrix(t, phi)  # [T x DM x DN]
        A = psi.reshape(-1, self.D * self.N)  # [TDM x DN]
        b = ref_y.reshape(A.shape[0], -1)  # [TDM x B]
        w = torch.inverse(A.transpose(0, 1).mm(A) + lambda_eps * torch.eye(A.shape[1])).mm(A.transpose(0, 1)).mm(b)
        return w

    def set_params_from_reference_trajectories(self, ref_y: torch.Tensor, t: Optional[torch.Tensor] = None,
                                               phi: Optional[torch.Tensor] = None, cov_eps=1e-7, fixed_cov=None,
                                               lambda_eps=1e-3):
        w = self.weights_from_trajectory(ref_y, t, phi, lambda_eps=lambda_eps)
        mu_w = w.mean(-1, keepdim=True)
        self.mu_w_params.data = mu_w.squeeze(-1)

        if fixed_cov is not None:
            n = self.N * self.D
            if self.is_cov_diagonal:
                self.cov_w_params.data = (fixed_cov * torch.ones(n)).sqrt()
            else:
                self.cov_w_params.data = (fixed_cov * torch.eye(n)).sqrt()
            return

        if self.is_cov_diagonal:
            self.cov_w_params.data = ((w - mu_w) ** 2).mean(-1).sqrt()
        else:
            v = w.transpose(0, 1).unsqueeze(-1) - mu_w
            cov_w = v.matmul(v.transpose(-2, -1)).mean(0)
            cov_w = (w.shape[-1] * cov_w + cov_eps * torch.eye(self.N * self.D)) / w.shape[-1]
            cov_w_params = torch.cholesky(cov_w, upper=False)
            self.cov_w_params.data = cov_w_params

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
        # b = b / torch.sum(b, -1, keepdim=True)
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
