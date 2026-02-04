import unittest
import numpy as np
from numpy.testing import assert_allclose
from BlochMcConnellSplitter import BlochMcConnellSplitter


class TestBlochMcConnellSplitter(unittest.TestCase):
    def setUp(self):
        self.R1 = np.ones((3,)) * 1e-9
        self.R2 = np.ones((3,)) * 1e-9
        self.gamma = 1.0
        self.k = np.zeros((3, 3))
        self.offset = np.zeros((3,))
        self.Meq = np.zeros(9)
        self.tau = 1e-5
        self.b1x = np.zeros(1000)
        self.b1y = np.zeros(1000)

    def create2pStatic(self):
        return BlochMcConnellSplitter(self.offset[:2], self.R1[:2], self.R2[:2], self.k[:2, :2],
                                      self.Meq[:6], self.gamma, self.tau, self.b1x, self.b1y)

    def create3pStatic(self):
        return BlochMcConnellSplitter(self.offset, self.R1, self.R2, self.k,
                                      self.Meq, self.gamma, self.tau, self.b1x, self.b1y)

    def test_noEvolution(self):
        static2pSplitter = self.create2pStatic()
        static3pSplitter = self.create3pStatic()
        M0 = np.arange(1, 10).astype(float)
        M = static2pSplitter.integrate(M0[:6])
        assert_allclose(M, M0[:6], rtol=1e-9, atol=0)
        M = static3pSplitter.integrate(M0)
        assert_allclose(M, M0, rtol=1e-9, atol=0)

    def test_transverseRelaxation(self):
        relaxing_R1 = np.array([1e-1, 3e-1, 1.0])
        relaxing_R2 = np.array([1e-3, 3e-3, 1e-2])
        relaxing2pSplitter = BlochMcConnellSplitter(self.offset[:2], relaxing_R1[:2], relaxing_R2[:2], self.k[:2, :2], self.Meq[:6], self.gamma, self.tau, self.b1x, self.b1y)
        M0 = np.arange(1, 10).astype(float)
        M = relaxing2pSplitter.integrate(M0[:6])
        # For 2-pool, transverse components are at indices 0 and 3
        assert_allclose(M[[0, 3]], np.exp(-1e-2 * relaxing_R2[:2]) * M0[[0, 3]], rtol=1e-9)
        assert_allclose(M[[2, 5]], np.exp(-1e-2 * relaxing_R1[:2]) * M0[[2, 5]], rtol=1e-9)
        relaxing3pSplitter = BlochMcConnellSplitter(self.offset, relaxing_R1, relaxing_R2, self.k, self.Meq, self.gamma, self.tau, self.b1x, self.b1y)
        M = relaxing3pSplitter.integrate(M0)
        assert_allclose(M[0::3], np.exp(-1e-2 * relaxing_R2) * M0[0::3], rtol=1e-9)
        assert_allclose(M[1::3], np.exp(-1e-2 * relaxing_R2) * M0[1::3], rtol=1e-9)
        assert_allclose(M[2::3], np.exp(-1e-2 * relaxing_R1) * M0[2::3], rtol=1e-9)

    def test_exchangeOnly(self):
        M0 = np.ones(9)
        exchanging_k = np.zeros((3, 3))
        exchanging_k[0, 1] = 1e3
        ex2 = BlochMcConnellSplitter(self.offset[:2], self.R1[:2], self.R2[:2], exchanging_k[:2, :2], self.Meq[:6], self.gamma, self.tau, self.b1x, self.b1y)
        M = ex2.integrate(M0[:6])
        t = 1e-2
        assert_allclose(M[0:3], np.exp(-exchanging_k[0,1] * t) * M0[0:3], rtol=1e-9)
        assert_allclose(M[3:6], (1 - np.exp(-exchanging_k[0,1] * t)) * M0[0:3] + M0[3:6], rtol=1e-9)
        ex3 = BlochMcConnellSplitter(self.offset, self.R1, self.R2, exchanging_k, self.Meq, self.gamma, self.tau, self.b1x, self.b1y)
        M = ex3.integrate(M0)
        assert_allclose(M[0:3], np.exp(-exchanging_k[0,1] * t) * M0[0:3], rtol=1e-9)
        assert_allclose(M[3:6], (1 - np.exp(-exchanging_k[0,1] * t)) * M0[0:3] + M0[3:6], rtol=1e-9)
        assert_allclose(M[6:9], M0[6:9], rtol=1e-9)

    def test_pulseOnly(self):
        M0 = np.zeros(9)
        M0[2::3] = np.array([1.0, 2.0, 3.0])
        pulse_length = 1000
        b1x = np.zeros(pulse_length)
        b1y = -np.ones(pulse_length) * np.pi / (2 * pulse_length * self.gamma * self.tau)
        pulsed2 = BlochMcConnellSplitter(self.offset[:2], self.R1[:2], self.R2[:2], self.k[:2, :2], self.Meq[:6], self.gamma, self.tau, b1x, b1y)
        M = pulsed2.integrate(M0[:6])
        assert_allclose(M[0::3], M0[2::3][:2], rtol=1e-9)
        assert_allclose(M[1::3], np.zeros(2), rtol=1e-9)
        assert_allclose(M[2::3], np.zeros(2), atol=1e-9)
        b1x = np.ones(pulse_length) * np.pi / (2 * self.gamma * pulse_length * self.tau)
        b1y = np.zeros(pulse_length)
        pulsed2.update_rotations(self.tau, b1x, b1y, self.gamma)
        M = pulsed2.integrate(M0[:6])
        assert_allclose(M[0::3], M0[0::3][:2], atol=1e-9)
        assert_allclose(M[1::3], M0[2::3][:2], rtol=1e-9)
        assert_allclose(M[2::3], np.zeros(2), atol=1e-9)
        # 3-pool: start with a y-axis 90-degree pulse (like setup in MATLAB tests)
        b1x = np.zeros(pulse_length)
        b1y = -np.ones(pulse_length) * np.pi / (2 * pulse_length * self.gamma * self.tau)
        pulsed3 = BlochMcConnellSplitter(self.offset, self.R1, self.R2, self.k, self.Meq, self.gamma, self.tau, b1x, b1y)
        M = pulsed3.integrate(M0)
        assert_allclose(M[0::3], M0[2::3], rtol=1e-9)
        assert_allclose(M[1::3], np.zeros(3), rtol=1e-9)
        assert_allclose(M[2::3], np.zeros(3), atol=1e-9)
        # Now test x-axis pulse
        b1x = np.ones(pulse_length) * np.pi / (2 * self.gamma * pulse_length * self.tau)
        b1y = np.zeros(pulse_length)
        pulsed3.update_rotations(self.tau, b1x, b1y, self.gamma)
        M = pulsed3.integrate(M0)
        assert_allclose(M[0::3], M0[0::3], atol=1e-9)
        assert_allclose(M[1::3], M0[2::3], rtol=1e-9)
        assert_allclose(M[2::3], np.zeros(3), atol=1e-9)


if __name__ == '__main__':
    unittest.main()

