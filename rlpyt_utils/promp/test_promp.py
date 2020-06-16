#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import unittest
from rlpyt_utils.promp.promp import ProMP
import torch


class TestProMP(unittest.TestCase):

    def test_psi_matrix_dim(self):
        promp = ProMP(2, num_basis_functions=10)
        psi_mat = promp.get_psi_matrix(torch.zeros(1))
        self.assertEqual(psi_mat.shape[0], 1)
        self.assertEqual(psi_mat.shape[1], 20)
        self.assertEqual(psi_mat.shape[2], 4)
        psi_mat = promp.get_psi_matrix(torch.zeros(100))
        self.assertEqual(psi_mat.shape[0], 100)
        self.assertEqual(psi_mat.shape[1], 20)
        self.assertEqual(psi_mat.shape[2], 4)

    def test_cov_y(self):
        promp = ProMP(2)
        t = torch.linspace(0, 1., 10)
        computed = promp.cov_y(t)
        psi_mat = promp.get_psi_matrix(t)
        expected = psi_mat.transpose(-2, -1).matmul(promp.cov_w).matmul(psi_mat)
        self.assertAlmostEqual(torch.sum((expected - computed) ** 2).detach().cpu().numpy(), 0.)

    def test_mu_y(self):
        promp = ProMP(2)
        t = torch.linspace(0, 1., 10)
        expected = promp.get_psi_matrix(t).transpose(-2, -1).matmul(promp.mu_w)
        computed = promp.mu_y(phi=promp.get_phi_tensor(t))
        self.assertAlmostEqual(torch.sum((expected - computed) ** 2).detach().cpu().numpy(), 0.)


if __name__ == '__main__':
    unittest.main()
