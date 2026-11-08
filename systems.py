import numpy as np

#2D charged particle functions

def B_2D(x):
    return np.array([0,0, np.sqrt(x[0]**2 + x[1]**2)])

def A_2D(x):
    A1 = -x[1] * np.sqrt(x[0]**2 + x[1]**2)
    A2 = x[0] * np.sqrt(x[0]**2 + x[1]**2)
    return np.array([A1, A2, 0])/3

def U_2D(x):
    return 1/(100*np.sqrt(x[0]**2 + x[1]**2))

def nabla_U_2D(x):
    u1 = x[0] / ((x[0]**2 + x[1]**2)**(3/2))
    u2 = x[1] / ((x[0]**2 + x[1]**2)**(3/2))
    u3 = 0
    return np.array([u1, u2, u3])/(-100)

x0_2D = np.array([0.0, 1.0, 0.1])
v0_2D = np.array([0.09, 0.05, 0.2])

#Tokamak magnetic field functions

def B_T(x):
    b1 = -(2*x[1] + x[0]*x[2]) / (2*(x[0]**2 + x[1]**2))
    b2 = (2*x[0] - x[1]*x[2]) / (2*(x[0]**2 + x[1]**2))
    b3 = (np.sqrt(x[0]**2 + x[1]**2) -1) / (2*np.sqrt(x[0]**2 + x[1]**2))
    return np.array([b1, b2, b3])

def A_T(x):
    a1 = 0
    a2 = 0.5 * (x[0] - np.log(np.sqrt(x[0]**2 + x[1]**2) + x[0]))
    a3 = -0.5 * (x[2] * np.arctan2(x[1], x[0]) + np.log(x[0]**2 + x[1]**2))
    return np.array([a1, a2, a3])

def U_T(x):
    return 0

def nabla_U_T(x):
    return np.zeros(3)

x0_T = np.array([1.2, 0.0, 0.0])
v0_T = np.array([0, 4.816e-4, -2.059e-3])