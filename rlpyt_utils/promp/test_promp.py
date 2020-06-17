#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import unittest
from rlpyt_utils.promp.promp import ProMP
import torch


class TestProMP(unittest.TestCase):

    def test_phi_dim(self):
        mpp = ProMP(2, num_basis_functions=15, position_only=True)
        mpv = ProMP(2, num_basis_functions=15)
        self.assertEqual(mpp.get_phi_tensor(torch.zeros(10)).shape, (10, 1, 15))
        self.assertEqual(mpv.get_phi_tensor(torch.zeros(10)).shape, (10, 2, 15))

    def test_psi_dim(self):
        mpp = ProMP(2, num_basis_functions=15, position_only=True)
        mpv = ProMP(2, num_basis_functions=15)
        self.assertEqual(mpp.get_psi_matrix(torch.zeros(10)).shape, (10, 2, 15 * 2))
        self.assertEqual(mpv.get_psi_matrix(torch.zeros(10)).shape, (10, 4, 15 * 2))

    def test_y_from_weights(self):
        mpp = ProMP(2, num_basis_functions=15, position_only=True)
        mpv = ProMP(2, num_basis_functions=15)

        t = torch.linspace(0, 1)
        w = torch.linspace(12, 25, 15 * 2)
        computed = mpp.y_from_weights(w, t)
        expected = mpp.get_psi_matrix(t).matmul(w)
        self.assertAlmostEqual(torch.sum((expected - computed) ** 2).detach().cpu().numpy(), 0.)
        computed = mpv.y_from_weights(w, t)
        expected = mpv.get_psi_matrix(t).matmul(w)
        self.assertAlmostEqual(torch.sum((expected - computed) ** 2).detach().cpu().numpy(), 0.)

    def test_mu_cov_y(self):
        mpp = ProMP(2, num_basis_functions=15, position_only=True)
        mpv = ProMP(2, num_basis_functions=15)

        t = torch.linspace(0, 1)
        mu, cov = mpp.mu_and_cov_y(t)
        psi = mpp.get_psi_matrix(t)
        psi_t = psi.transpose(-2, -1)
        mu_w, cov_w = mpp.mu_and_cov_w
        emu, ecov = psi.matmul(mu_w), psi.matmul(cov_w).matmul(psi_t) + mpp.sigma_y
        self.assertAlmostEqual(torch.sum((emu - mu) ** 2).detach().cpu().numpy(), 0.)
        self.assertAlmostEqual(torch.sum((ecov - cov) ** 2).detach().cpu().numpy(), 0.)

        mu, cov = mpv.mu_and_cov_y(t)
        psi = mpv.get_psi_matrix(t)
        psi_t = psi.transpose(-2, -1)
        mu_w, cov_w = mpv.mu_and_cov_w
        emu, ecov = psi.matmul(mu_w), psi.matmul(cov_w).matmul(psi_t) + mpv.sigma_y
        self.assertAlmostEqual(torch.sum((emu - mu) ** 2).detach().cpu().numpy(), 0.)
        self.assertAlmostEqual(torch.sum((ecov - cov) ** 2).detach().cpu().numpy(), 0.)


if __name__ == '__main__':
    unittest.main()
