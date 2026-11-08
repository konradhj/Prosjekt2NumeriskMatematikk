from systems import B_2D, A_2D, U_2D, nabla_U_2D, B_T, A_T, U_T, nabla_U_T
#from boris import run_boris
from RK4 import RK4, f
from symmetric_multistep import symmetric_multistep
from scipy.integrate import solve_ivp
import numpy as np
import time
import matplotlib.pyplot as plt

func_list = [[B_2D, A_2D, U_2D, nabla_U_2D], [B_T, A_T, U_T, nabla_U_T]]
h = 0.1
T = 100
x0_2D = np.array([0.0, 1.0, 0.1])
v0_2D = np.array([0.09, 0.05, 0.2])
x0_T = np.array([1.2, 0.0, 0.0])
v0_T = np.array([0, 4.816e-4, -2.059e-3])
x0_list = [x0_2D, x0_T]
v0_list = [v0_2D, v0_T]
dimension_names = ["2D Charged Particle", "Tokamak"]

h_values = []

rk4_errors = []
boris_errors = []
multi_errors = []

rk4_times = []
boris_times = []
multi_times = []

for dim in range(1):
    funcs = func_list[dim]
    B, A, U, gradU = funcs
    x0 = x0_list[dim]
    v0 = v0_list[dim]

    def rhs(t, y):
        x = y[:3]
        v = y[3:]
        return np.concatenate((v, f(x, v, B, gradU)))

    for i in range(0, 3):
        h_i = h/ (2)**i
        N_i = int(T / h_i)

        y0 = np.concatenate((x0, v0))
        ref = solve_ivp(rhs,(0, T),y0,method="RK45",rtol=1e-12,atol=1e-12)
        x_ref = ref.y[:3, -1]

        start = time.perf_counter()
        x_rk4, v_rk4 = RK4(x0, v0, B, gradU, h_i, T)
        rk4_times.append(time.perf_counter() - start)

        """
        start = time.perf_counter()
        boris_solve = run_boris(x0, v0, B, gradU, h_i, T)
        boris_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        multi_solve = 
        multi_times.append(time.perf_counter() - start)
        """

        rk4_error = np.linalg.norm(x_rk4[-1] - x_ref)
        #boris_error = np.linalg.norm(x_boris[-1] - x_ref)
        #multi_error = np.linalg.norm(x_multi[-1] - x_ref)

        rk4_errors.append(rk4_error)
        #boris_errors.append(boris_error)
        #multi_errors.append(multi_error)

        h_values.append(h_i)

    h_values = np.array(h_values)
    plt.title(f"Order verification for " + dimension_names[dim])
    plt.loglog(h_values, rk4_errors, 'o-', label='RK4')
    #plt.loglog(h_values, boris_errors, 'o-', label='Boris')
    #plt.loglog(h_values, multi_errors, 'o-', label='Multistep')

    #C = boris_errors[0] / h_values[0]**2
    D = rk4_errors[0] / h_values[0]**4
    #plt.loglog(h_values, C*h_values**2, '--', label='$h^2$')
    plt.loglog(h_values, D*h_values**4, '--', label='$h^4$')
    plt.xlabel("Step size h")
    ax = plt.gca()
    ax.set_xticks(h_values)
    ax.set_xticklabels(['0.1', '0.05', '0.025'])
    plt.ylabel("Global error")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.show()

    plt.title(f"Work–precision diagram " + dimension_names[dim])
    plt.loglog(rk4_errors, rk4_times, 'o-', label='RK4')
    #plt.loglog(boris_errors, boris_times, 'o-', label='Boris')
    #plt.loglog(multi_errors, multi_times, 'o-', label='Multistep')
    plt.xlabel("Global error")
    plt.ylabel("Computing time (s)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


