import numpy as np


def boris_step(x, v_half, h, B, grad_U):
    """Advance (x_n, v_{n-1/2}) to (x_{n+1}, v_{n+1/2}).

    Inputs are not modified. The velocity input must already be staggered.
    """
    x = np.asarray(x, dtype=float)
    v_half = np.asarray(v_half, dtype=float)
    kick = 0.5 * h * np.asarray(grad_U(x), dtype=float)
    t = 0.5 * h * np.asarray(B(x), dtype=float)
    s = 2 * t / (1 + np.dot(t, t))

    v_minus = v_half - kick
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v_next_half = v_plus - kick
    return x + h * v_next_half, v_next_half

