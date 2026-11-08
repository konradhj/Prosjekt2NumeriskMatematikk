"""5C: check the symmetry of the method in Section 4.3.

Run with: python multistep_symmetry.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from systems import (
    A_2D, B_2D, nabla_U_2D, x0_2D, v0_2D,
    A_T, B_T, nabla_U_T, x0_T, v0_T,
)


def jac_A_2D(x):
    """Jacobian of A: entry (i, j) is dA_i/dx_j."""
    a, b, _ = x
    r = np.hypot(a, b)
    return np.array([
        [-a*b/r, -r-b*b/r, 0],
        [r+a*a/r, a*b/r, 0],
        [0, 0, 0],
    ]) / 3


def jac_A_T(x):
    a, b, c = x
    r = np.hypot(a, b)
    return np.array([
        [0, 0, 0],
        [0.5*(1-1/r), -b/(2*r*(r+a)), 0],
        [(0.5*c*b-a)/r**2, (-0.5*c*a-b)/r**2,
         -0.5*np.arctan2(b, a)],
    ])


def derivative(y, j, h):
    """Five-point derivative at index j, with signed step size h."""
    return (y[j-2] - 8*y[j-1] + 8*y[j+1] - y[j+2]) / (12*h)


def multistep(start, h, N, A, jac_A, grad_U):
    """Return N+1 positions, given the first eight positions in start."""
    alpha = np.array([1, -7/5, 9/25, 22/125, -34/125,
                      22/125, 9/25, -7/5, 1])
    a1, a2, a3 = -0.7, 0.1, 0.9
    product = a1*a2*a3
    pairs = a1*a2 + a1*a3 + a2*a3
    total = a1 + a2 + a3
    beta0 = (20*product - 4*pairs - 28*total - 52) / 3
    beta1 = (2*product + 14*pairs + 26*total + 38) / 3

    x = np.zeros((N+1, 3))
    x[:8] = start
    for k in range(8, N+1):
        history = x[k-8:k]
        potentials = np.array([A(position) for position in history])
        forces = []
        # Indices 3, 4, 5 correspond to F_{n-1}, F_n, F_{n+1}.
        for j in (3, 4, 5):
            v = derivative(history, j, h)
            dA = derivative(potentials, j, h)
            forces.append(jac_A(history[j]).T @ v - dA - grad_U(history[j]))
        rhs = h**2 * (beta1*forces[0] + beta0*forces[1] + beta1*forces[2])
        x[k] = rhs - alpha[:8] @ history  # Equation (10), alpha_4 = 1.
    return x


def main():
    h, T = 0.1, 20.0
    N = round(T/h)
    time = h * np.arange(N+1)
    systems = [
        ("2D particle", x0_2D, v0_2D, B_2D, A_2D, jac_A_2D, nabla_U_2D),
        ("Tokamak", x0_T, v0_T, B_T, A_T, jac_A_T, nabla_U_T),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (name, x0, v0, B, A, jac_A, grad_U) in zip(axes, systems):
        def rhs(t, y):
            return np.concatenate((y[3:], np.cross(y[3:], B(y[:3])) - grad_U(y[:3])))

        # Generate only the eight starting positions with solve_ivp.
        initial = solve_ivp(rhs, (0, time[7]), np.concatenate((x0, v0)),
                            t_eval=time[:8], rtol=1e-12, atol=1e-14)
        if not initial.success:
            raise RuntimeError(initial.message)
        forward = multistep(initial.y[:3].T, h, N, A, jac_A, grad_U)

        # Reuse the final history; do not generate new starting values.
        backward = multistep(forward[-8:][::-1], -h, N, A, jac_A, grad_U)
        recovered = backward[::-1]  # Align both arrays by physical time.
        error = np.linalg.norm(recovered - forward, axis=1)

        print(f"{name}: return error at t=0 = {error[0]:.3e}")
        print(f"  Maximum difference along the path = {error.max():.3e}")
        # The final eight positions were copied, so exclude them from the plot.
        ax.plot(time[:-8], error[:-8])
        ax.set(title=name, xlabel="Physical time", ylabel="Forward/backward position difference")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.3)

    fig.suptitle("Symmetry check: differences should be near rounding error")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
