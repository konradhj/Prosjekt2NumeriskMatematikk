"""5C: same speed, different launch angles in a magnetic bottle.

Run with: python magnetic_bottle.py
"""

import matplotlib.pyplot as plt
import numpy as np

from RK4 import RK4


L = 5.0  # Bottle ends at z = -L and z = L.


def B_bottle(x):
    """Smooth, divergence-free model field, strongest at the bottle ends."""
    z = x[2]
    b = 1 + 3 * np.sin(np.pi * z / (2 * L))**2
    db_dz = 3 * np.pi / (2 * L) * np.sin(np.pi * z / L)
    # These transverse components ensure div(B) = 0.
    return np.array([-0.5 * x[0] * db_dz, -0.5 * x[1] * db_dz, b])


def grad_U(x):
    return np.zeros(3)  # No electric force.


def main(h=0.02):
    T = 200.0
    speed = 0.2
    angles = [20, 28, 32, 60]  # Degrees from the magnetic field at the centre.
    x0 = np.zeros(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Near-axis mirror theory: sin(theta_c)^2 = B_min / B_max = 1/4.
    # Approximate prediction; assumes the field varies slowly over a gyration.
    # https://farside.ph.utexas.edu/teaching/plasma/Plasma/node22.html
    print("Approximate theoretical transition: 30 degrees")

    for angle in angles:
        theta = np.radians(angle)
        v0 = speed * np.array([np.sin(theta), 0, np.cos(theta)])
        x, v = RK4(x0, v0, B_bottle, grad_U, h, T)
        time = h * np.arange(len(x))

        # Escape means leaving this bottle through either end. Ignore the
        # subsequent motion: the model field repeats outside this region.
        crossings = np.flatnonzero(np.abs(x[:, 2]) >= L)
        if len(crossings):
            stop = crossings[0] + 1
            status = f"escaped at t = {time[stop - 1]:.2f}"
        else:
            stop = len(x)
            status = f"remained inside until t = {T:g}"

        energy = 0.5 * np.sum(v[:stop]**2, axis=1)
        relative_error = np.max(np.abs(energy - energy[0])) / energy[0]
        print(f"{angle:2d} degrees: {status}; max relative energy error = {relative_error:.2e}")

        axes[0].plot(time[:stop], x[:stop, 2], label=f"{angle}°")
        axes[1].plot(x[:stop, 2], x[:stop, 0], label=f"{angle}°")

    for end in (-L, L):
        axes[0].axhline(end, color="grey", linestyle="--")
        axes[1].axvline(end, color="grey", linestyle="--")
    axes[0].set(xlabel="Time", ylabel="z", title="Bouncing or escaping?")
    axes[1].set(xlabel="z", ylabel="x", title="Trajectory projection")
    for ax in axes:
        ax.legend(title="Launch angle")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
