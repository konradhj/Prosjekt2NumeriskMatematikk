import numpy as np
from scipy.integrate import solve_ivp

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

def run_boris(x0, v0, B, nabla_U, h, T):
    N = int(round(T/h))
    x = np.zeros((N+1, 3))
    v = np.zeros_like(x)
    x[0] = x0

    # Boris trenger hastigheten ved t=-h/2 som startverdi.
    def rhs(t, y):
        return np.concatenate((y[3:], np.cross(y[3:], B(y[:3])) - nabla_U(y[:3])))
    start = solve_ivp(rhs, [0, -h/2], np.concatenate((x0, v0)),
                      rtol=1e-10, atol=1e-12)
    if not start.success:
        raise RuntimeError(start.message)
    v_half = start.y[3:, -1]

    for i in range(N):
        x[i+1], v_next_half = boris_step(x[i], v_half, h, B, nabla_U)
        v[i] = (v_half + v_next_half)/2
        v_half = v_next_half

    # Ett ekstra halvsteg gir også hastigheten ved siste tidspunkt.
    _, v_next_half = boris_step(x[-1], v_half, h, B, nabla_U)
    v[-1] = (v_half + v_next_half)/2
    v[0] = v0
    return x, v

