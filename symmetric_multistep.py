"""Symmetrisk flerstegsmetode av orden 4 fra prosjektet, oppgave 4.3."""

import numpy as np
from scipy.integrate import solve_ivp


def symmetric_multistep(B, A, jac_A, grad_U, x0, v0, T, h):
    """Returner tidspunktene t og posisjonene x, med x[k] ≈ x(t[k]).

    Modellfunksjonene tar en posisjonsvektor med tre komponenter.
    jac_A(x)[i, j] = d A_i / d x_j. De øvrige returnerer 3-vektorer.
    T må være et helt antall steg h, og minst 7*h.
    """
    if not np.isfinite(h) or not np.isfinite(T) or h <= 0 or T <= 0:
        raise ValueError("h og T må være positive og endelige.")
    N = int(round(T / h))
    if N < 7 or not np.isclose(N * h, T, rtol=1e-12, atol=1e-14):
        raise ValueError("T må være et helt antall steg h, og minst 7*h.")

    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if x0.shape != (3,) or v0.shape != (3,):
        raise ValueError("x0 og v0 må ha tre komponenter hver.")

    t = h * np.arange(N + 1)
    x = np.empty((N + 1, 3))
    A_values = np.empty_like(x)

    # Startverdiene x[0], ..., x[7] fra ligning (1).
    def rhs(time, y):
        position, velocity = y[:3], y[3:]
        acceleration = np.cross(velocity, B(position)) - grad_U(position)
        return np.concatenate((velocity, acceleration))

    start = solve_ivp(
        rhs, (t[0], t[7]), np.concatenate((x0, v0)),
        method="RK45", t_eval=t[:8], rtol=1e-10, atol=1e-12,
    )
    if not start.success:
        raise RuntimeError(f"Kunne ikke beregne startverdier: {start.message}")
    x[:8] = start.y[:3].T
    for k in range(8):
        A_values[k] = A(x[k])

    a1, a2, a3 = -0.7, 0.1, 0.9
    product = a1 * a2 * a3
    pairs = a1 * a2 + a1 * a3 + a2 * a3
    total = a1 + a2 + a3
    beta0 = (20 * product - 4 * pairs - 28 * total - 52) / 3
    beta1 = (2 * product + 14 * pairs + 26 * total + 38) / 3
    alpha = np.array([
        1, -7/5, 9/25, 22/125, -34/125, 22/125, 9/25, -7/5, 1
    ])

    def derivative(values, j):
        # Delta(values[j]) / h: fempunkts sentraldifferanse.
        return (
            values[j - 2] - 8 * values[j - 1]
            + 8 * values[j + 1] - values[j + 2]
        ) / (12 * h)

    def F(j):
        velocity = derivative(x, j)
        A_delta = derivative(A_values, j)
        return np.asarray(jac_A(x[j])).T @ velocity - A_delta - grad_U(x[j])

    # Gjenbruk F-verdiene når sentrum n flyttes ett steg frem.
    F_previous, F_current = F(3), F(4)
    for n in range(4, N - 3):
        F_next = F(n + 1)
        right_side = h**2 * (
            beta1 * F_previous + beta0 * F_current + beta1 * F_next
        )
        # Isoler x[n+4] i ligning (10). alpha[8] = alpha_4 = 1.
        x[n + 4] = (right_side - alpha[:8] @ x[n - 4:n + 4]) / alpha[8]
        A_values[n + 4] = A(x[n + 4])
        F_previous, F_current = F_current, F_next

    return t, x
