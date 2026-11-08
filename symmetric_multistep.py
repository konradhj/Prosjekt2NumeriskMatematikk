import numpy as np
from scipy.integrate import solve_ivp


def symmetric_multistep(x0, v0, B, A, jac_A, nabla_U, h, T):
    """Beregn tidspunktene t og posisjonene x."""
    N = int(round(T / h))
    if N < 7 or not np.isclose(N*h, T):
        raise ValueError("T må være minst 7 steg og delelig med h.")

    t = np.arange(N + 1) * h
    x = np.zeros((N + 1, 3))
    A_values = np.zeros_like(x)

    # Beregn de åtte startposisjonene.
    def rhs(t, y):
        position = y[:3]
        velocity = y[3:]
        acceleration = np.cross(velocity, B(position)) - nabla_U(position)
        return np.concatenate((velocity, acceleration))

    start = solve_ivp(
        rhs, [0, t[7]], np.concatenate((x0, v0)),
        t_eval=t[:8], rtol=1e-10, atol=1e-12
    )
    if not start.success:
        raise RuntimeError(start.message)

    x[:8] = start.y[:3].T
    for i in range(8):
        A_values[i] = A(x[i])

    # Koeffisientene for a1=-0.7, a2=0.1 og a3=0.9.
    alpha = np.array([
        1, -7/5, 9/25, 22/125, -34/125,
        22/125, 9/25, -7/5, 1
    ])
    beta0 = -19.74
    beta1 = 12.378

    # Fempunkts sentraldifferanse.
    def derivative(values, j):
        return (
            values[j-2] - 8*values[j-1]
            + 8*values[j+1] - values[j+2]
        ) / (12*h)

    def F(j):
        velocity = derivative(x, j)
        A_derivative = derivative(A_values, j)
        return jac_A(x[j]).T @ velocity - A_derivative - nabla_U(x[j])

    F_previous = F(3)
    F_current = F(4)

    for n in range(4, N-3):
        F_next = F(n+1)

        right_side = h**2 * (
            beta1*F_previous + beta0*F_current + beta1*F_next
        )

        # Isoler neste posisjon; alpha_4 = 1.
        x[n+4] = right_side - alpha[:8] @ x[n-4:n+4]
        A_values[n+4] = A(x[n+4])

        F_previous, F_current = F_current, F_next

    return t, x

"""Defnerer Jacobi-matrisene"""
def jac_A_2D(x):
    x1, x2, _ = x
    r = np.sqrt(x1**2 + x2**2)

    return np.array([
        [-x1*x2/r,    -r - x2**2/r, 0],
        [r + x1**2/r,  x1*x2/r,    0],
        [0,           0,           0]
    ]) / 3


def jac_A_T(x):
    x1, x2, x3 = x
    r2 = x1**2 + x2**2
    r = np.sqrt(r2)

    return np.array([
        [0, 0, 0],
        [(1 - 1/r)/2, -x2/(2*r*(r + x1)), 0],
        [(x3*x2 - 2*x1)/(2*r2),
         -(x3*x1 + 2*x2)/(2*r2),
         -np.arctan2(x2, x1)/2]
    ])