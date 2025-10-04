import numpy as np


class Basis1D:
    def __init__(self, N, axis=None):
        self.N = N
        j = np.arange(1, N + 1)  # cria um array com j = 1, 2, ..., N
        self.x = j * np.pi / (N + 1)  # calcula os pontos x_j
        self.r = self.x

        self.axis = axis

        self.jacobian = 1.0
        if axis is not None:
            a = axis[0]
            b = axis[1]

            self.r = (b - a) * self.x / (np.pi) + a
            self.jacobian = np.pi / (b - a)

        self.Cx = self.Basis(N)
        self.Cx_inv = np.linalg.inv(self.Cx)

        self.Cx_xx = self.jacobian**2 * self.Basis_xx(N)

        self.D2 = self.jacobian**2 * self.differentiation_matrix()

    def dct(self, u):
        u_hat = u @ self.Cx_inv
        return u_hat

    def idct(self, u_hat):
        u = u_hat @ self.Cx
        return u

    def Basis(self, n):
        Cx = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                Cx[i, j] = np.sin((i + 1) * self.x[j])

        return Cx

    def Basis_xx(self, n):
        Cx = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                Cx[i, j] = -(i+1)**2 * np.sin((i+1) * self.x[j])

        return Cx

    def differentiation_matrix(self):
        D2 = np.zeros((self.N, self.N))

        for i in range(self.N):
            D2[i, i] = -((i + 1) ** 2)

        return D2
    

# Rk4 algorithm
def rk4(u, v, dt, f, g):
    k1 = dt * f(u, v)
    l1 = dt * g(u, v)

    k2 = dt * f(u + 0.5 * k1, v + 0.5 * l1)
    l2 = dt * g(u + 0.5 * k1, v + 0.5 * l1)

    k3 = dt * f(u + 0.5 * k2, v + 0.5 * l2)
    l3 = dt * g(u + 0.5 * k2, v + 0.5 * l2)

    k4 = dt * f(u + k3, v + l3)
    l4 = dt * g(u + k3, v + l3)

    u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    v_next = v + (l1 + 2*l2 + 2*l3 + l4) / 6

    return u_next, v_next



