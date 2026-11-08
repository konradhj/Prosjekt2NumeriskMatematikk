# Legg denne filen i samme mappe som de fire metode- og modellfilene.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import systems as s
from RK4 import RK4
from boris import boris_step
from boris import run_boris
from symmetric_multistep import symmetric_multistep, jac_A_2D, jac_A_T

# 5a: kjør metodene og lag plottene for begge modellene.
T = 1e5
h = 0.1

models = [
    ("2D", s.B_2D, s.A_2D, jac_A_2D, s.U_2D, s.nabla_U_2D, s.x0_2D, s.v0_2D),
    ("Tokamak", s.B_T, s.A_T, jac_A_T, s.U_T, s.nabla_U_T, s.x0_T, s.v0_T),
]

for model, B, A, J, U, G, x0, v0 in models:
    print("Beregner", model)
    t = np.arange(int(round(T/h))+1)*h

    print("  RK4", flush=True)
    x_rk, v_rk = RK4(x0, v0, B, G, h, T)
    print("  Boris", flush=True)
    x_boris, v_boris = run_boris(x0, v0, B, G, h, T)
    print("  Symmetrisk flersteg", flush=True)
    t_sym, x_sym = symmetric_multistep(x0, v0, B, A, J, G, h, T)
    assert len(x_rk) == len(t) and np.allclose(t_sym, t)

    # Samme referanseløsning brukes til alle tre metodene.
    def rhs(t, y):
        return np.concatenate((y[3:], np.cross(y[3:], B(y[:3])) - G(y[:3])))
    print("  Referanseløsning", flush=True)
    ref = solve_ivp(rhs, [0, T], np.concatenate((x0, v0)),
                    t_eval=t, rtol=1e-12, atol=1e-14)
    if not ref.success:
        raise RuntimeError(ref.message)
    x_ref = ref.y[:3].T

    # Flerstegsmetodens hastigheter er definert ved t[2:-2].
    v_sym = (x_sym[:-4] - 8*x_sym[1:-3]
             + 8*x_sym[3:-1] - x_sym[4:])/(12*h)
    E0 = np.sum(v0**2)/2 + U(x0)
    methods = [("RK4", x_rk, v_rk), ("Boris", x_boris, v_boris),
               ("Symmetrisk flersteg", x_sym, v_sym)]

    colors = ["C0", "C1", "C2"]

    # Ett banepanel per metode: kurvene skjuler ikke hverandre.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for axis, (name, x, v), color in zip(axes, methods, colors):
        axis.plot(x[:, 0], x[:, 1], color=color, linewidth=0.6)
        axis.set(xlabel="$x_1$", ylabel="$x_2$", title=name)
        if model == "2D":
            axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.3)
    fig.suptitle(f"{model}: baner i $(x_1,x_2)$, T={T:g}, h={h:g}")
    fig.tight_layout()

    # Hele posisjonsfeilen i tre dimensjoner, ved de samme tidspunktene.
    fig, axis = plt.subplots(figsize=(8, 4))
    for (name, x, v), color in zip(methods, colors):
        error = np.linalg.norm(x-x_ref, axis=1)
        axis.semilogy(t, error, color=color, label=name, nonpositive="mask")
    axis.set(xlabel="$t$", ylabel="$\\|x_n-x_{ref}(t_n)\\|_2$",
             title=f"{model}: posisjonsfeil, T={T:g}, h={h:g}")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()

    # Samme energidefinisjon og samme y-skala for alle metodene.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for axis, (name, x, v), color in zip(axes, methods, colors):
        if name == "Symmetrisk flersteg":
            energy_t, energy_x = t[2:-2], x[2:-2]
        else:
            energy_t, energy_x = t, x
        E = np.sum(v**2, axis=1)/2 + np.array([U(p) for p in energy_x])
        energy_error = np.abs(E-E0)
        axis.semilogy(energy_t, energy_error, color=color, linewidth=0.5,
                      alpha=0.35, label="Alle steg", nonpositive="mask")

        # Ekstra hjelpelinje: maksfeil i tidsblokker, ikke en ny energidefinisjon.
        block = max(1, len(energy_error)//200)
        starts = np.arange(0, len(energy_error), block)
        maxima = [np.max(energy_error[i:i+block]) for i in starts]
        ends = np.minimum(starts+block-1, len(energy_error)-1)
        axis.semilogy(energy_t[ends], maxima, color=color,
                      label="Blokkmaksima", nonpositive="mask")
        axis.set(xlabel="$t$", title=name)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("$|E_n-E(x_0,v_0)|$")
    fig.suptitle(f"{model}: energifeil, T={T:g}, h={h:g}")
    fig.tight_layout()

    # Tokamak: R på x-aksen og x3 på y-aksen, som i oppgaven.
    if model == "Tokamak":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for axis, (name, x, v), color in zip(axes, methods, colors):
            R = np.sqrt(x[:, 0]**2+x[:, 1]**2)
            axis.plot(R, x[:, 2], color=color, linewidth=0.6)
            axis.set(xlabel="$R=\\sqrt{x_1^2+x_2^2}$", title=name)
            axis.grid(alpha=0.3)
        axes[0].set_ylabel("$x_3$")
        fig.suptitle(f"Tokamak: $(R,x_3)$-baner, T={T:g}, h={h:g}")
        fig.tight_layout()

    print("  Ferdig. Lukk figurvinduene for å fortsette.", flush=True)
    plt.show()
