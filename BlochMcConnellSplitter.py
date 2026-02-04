import numpy as np

class BlochMcConnellSplitter:
    """Python translation of the MATLAB BlochMcConnellSplitter class.

    Usage:
        s = BlochMcConnellSplitter(offset, R1, R2, k, Meq, gamma, tau, b1x, b1y)
        M = s.integrate(M0)
    """

    def __init__(self, offset, R1, R2, k, Meq, gamma, tau, b1x, b1y):
        self.n_pools = int(np.asarray(offset).shape[0])
        self.tau = tau
        self.update_rotations(tau, b1x, b1y, gamma)
        self.update_relaxation_exchange(offset, R1, R2, k, Meq)

    def integrate(self, M0):
        """Integrate using asymmetric operator splitting.
        M0: 1D array of length 3*n_pools
        returns 1D array of length 3*n_pools
        """
        M = np.asarray(M0).reshape(1, -1).astype(np.complex128)
        num_steps = self.g_rot.shape[0]
        for i in range(num_steps):
            M = M @ self.g_rot[i]
            M = (M + self.RC) @ self.expRt - self.RC
        return np.real(M.T).ravel()

    def update_relaxation_exchange(self, offset, R1, R2, k, Meq):
        R = (self.relaxation_matrix(R1, R2) +
             self.exchange_matrix(k) +
             self.offset_matrix(offset))
        # Form C vector: zeros for transverse entries, R1 for longitudinal
        n = self.n_pools
        c = np.zeros(3 * n)
        for i in range(n):
            c[3 * i + 2] = R1[i]
        C = Meq * c
        # Solve R x = C
        x = np.linalg.solve(R, C)
        self.RC = x.reshape(1, -1)
        # Compute matrix exponential via diagonalization
        D, V = np.linalg.eig(R)
        expD = np.diag(np.exp(D * self.tau))
        expR = V @ expD @ np.linalg.inv(V)
        # Match MATLAB's transposed storage used in multiply M * expRt
        self.expRt = expR.T

    def update_rotations(self, tau, b1x, b1y, gamma):
        self.g_rot = self.create_rotations(self.n_pools, tau, np.asarray(b1x).ravel(), np.asarray(b1y).ravel(), gamma)

    @staticmethod
    def create_rotations(n_pools, tau, b1x, b1y, gamma):
        # b1x, b1y are arrays of length N (timesteps)
        b1x = np.asarray(b1x).ravel()
        b1y = np.asarray(b1y).ravel()
        N = b1x.size
        phi = -gamma * tau * np.abs(b1x - 1j * b1y)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        n1 = gamma * tau * b1x
        n2 = gamma * tau * b1y
        n3 = np.zeros_like(b1x)
        nonzero = np.abs(phi) > 0
        # avoid divide by zero
        denom = np.abs(phi)
        denom[~nonzero] = 1.0
        n1 = np.where(nonzero, n1 / denom, 0.0)
        n2 = np.where(nonzero, n2 / denom, 0.0)

        # Build 3x3 rotation matrices for each timestep
        rot3 = np.zeros((N, 3, 3), dtype=np.complex128)
        for i in range(N):
            n_vec = np.array([n1[i], n2[i], n3[i]])
            outer = np.outer(n_vec, n_vec)
            rot = (1 - cos_phi[i]) * outer + np.array([
                [cos_phi[i], -n3[i] * sin_phi[i], n2[i] * sin_phi[i]],
                [n3[i] * sin_phi[i], cos_phi[i], -n1[i] * sin_phi[i]],
                [-n2[i] * sin_phi[i], n1[i] * sin_phi[i], cos_phi[i]]
            ], dtype=np.complex128)
            # Transpose so block diagonal matches MATLAB permute(temp,[2,1,3]) behavior
            rot3[i] = rot.T

        # Create block-diagonal rotation for n_pools by Kron product
        big_rot = np.zeros((N, 3 * n_pools, 3 * n_pools), dtype=np.complex128)
        I = np.eye(n_pools)
        for i in range(N):
            big_rot[i] = np.kron(I, rot3[i])

        return big_rot

    @staticmethod
    def relaxation_matrix(R1, R2):
        n = int(np.asarray(R1).shape[0])
        n_dim = 3 * n
        R = np.zeros((n_dim, n_dim))
        for i in range(n):
            R[3 * i + 0, 3 * i + 0] = -R2[i]
            R[3 * i + 1, 3 * i + 1] = -R2[i]
            R[3 * i + 2, 3 * i + 2] = -R1[i]
        return R

    @staticmethod
    def offset_matrix(offset):
        # build block diagonal of 3x3 blocks
        blocks = []
        for off in np.asarray(offset).ravel():
            b = np.zeros((3, 3))
            b[1, 0] = off
            b[0, 1] = -off
            blocks.append(b)
        # block diagonal via kron
        n = len(blocks)
        M = np.zeros((3 * n, 3 * n))
        for i, b in enumerate(blocks):
            M[3 * i:3 * i + 3, 3 * i:3 * i + 3] = b
        return M

    @staticmethod
    def exchange_matrix(k):
        k = np.asarray(k)
        if k.shape[0] == 2:
            offdiag = np.block([
                [np.zeros((3, 3)), k[1, 0] * np.eye(3)],
                [k[0, 1] * np.eye(3), np.zeros((3, 3))]
            ])
            diagonal = np.block([
                [(-np.sum(k[0, :]) + k[0, 0]) * np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), (-np.sum(k[1, :]) + k[1, 1]) * np.eye(3)]
            ])
        elif k.shape[0] == 3:
            offdiag = np.block([
                [np.zeros((3, 3)), k[1, 0] * np.eye(3), k[2, 0] * np.eye(3)],
                [k[0, 1] * np.eye(3), np.zeros((3, 3)), k[2, 1] * np.eye(3)],
                [k[0, 2] * np.eye(3), k[1, 2] * np.eye(3), np.zeros((3, 3))]
            ])
            diagonal = np.block([
                [(-np.sum(k[0, :]) + k[0, 0]) * np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((3, 3)), (-np.sum(k[1, :]) + k[1, 1]) * np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), (-np.sum(k[2, :]) + k[2, 2]) * np.eye(3)]
            ])
        else:
            raise ValueError("exchange_matrix only supports 2x2 or 3x3 k-matrices")
        return diagonal + offdiag
