import numpy as np

def RK4(x0, v0, B, nabla_U, h, T):
    N = int(T/h)
    x = np.zeros((N+1, 3))
    v = np.zeros((N+1, 3))
    x[0] = x0
    v[0] = v0
    def f(x,v):
        return np.cross(v, B(x)) - nabla_U(x)

    for i in range(N):
        kv1 = h * f(x[i], v[i])
        kx1 = h * v[i]

        kv2 = h * f(x[i] + 0.5*kx1, v[i] + 0.5*kv1)
        kx2 = h * (v[i] + 0.5*kv1)

        kv3 = h * f(x[i] + 0.5 * kx2, v[i] + 0.5 * kv2)
        kx3 = h * (v[i] + 0.5 * kv2)

        kv4 = h * f(x[i] + kx3, v[i] + kv3)
        kx4 = h * (v[i] + kv3)

        v_next = v[i] + (kv1 + 2*kv2 + 2*kv3 + kv4)/6
        x_next = x[i] + (kx1 + 2*kx2 + 2*kx3 + kx4)/6

        v[i+1] = v_next
        x[i+1] = x_next
    return x, v