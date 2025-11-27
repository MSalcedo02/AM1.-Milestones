import numpy as np
import matplotlib.pyplot as plt
from numpy import concatenate, array, zeros, abs, linspace
from numpy.linalg import norm, solve, LinAlgError
from matplotlib import pyplot as plt


#   HERRAMIENTAS A UTILIZAR: DERIVADA, JACOBIANO, NEWTON


def derivada(f, x, dx):
    return (f(x + dx) - f(x - dx)) / (2 * np.linalg.norm(dx))

def Jacobiano(f, x):
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = 1e-7
        J[:, j] = derivada(f, x, dx)
    return J

def Gauss(A, b):
    return solve(A, b)

def Newton(f, x0, tol=1e-10, max_iter=40):
    x = x0
    Dx = 1.

    for k in range(max_iter):
        if norm(Dx) < tol:
            break
        J = Jacobiano(f, x)
        Dx = Gauss(J, - f(x))
        x = x + Dx

    return x



#   Esquemas temporales: Euler, Crank-Nicholson, Runge-Kutta 4


def Euler(U1, t1, t2, F):
    dt = t2 - t1 
    return U1 + dt * F(U1, t1)

# def Inverse_Euler(U, t1, t2, F):
#     dt = t2 - t1
#     def G(x):
#         return x - U - dt * F(x, t2)
#     return Newton(G, U)

def Crank_Nicolson(U1, t1, t2, F):
    dt = t2 - t1
    a = U1 + dt/2 * F(U1, t1)
    def G(x):
        return x - a - dt/2 * F(x, t2)
    return Newton(G, U1)

def RK4(U1, t1, t2, F):
    dt = t2 - t1
    k1 = F(U1, t1)
    k2 = F(U1 + 0.5 * dt * k1, t1 + 0.5 * dt)
    k3 = F(U1 + 0.5 * dt * k2, t1 + 0.5 * dt)
    k4 = F(U1 + dt * k3, t1 + dt)
    return U1 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)



#   PROBLEMA N-CUERPOS


def N_body_problem(U, t, masses, G=1.0):

    N_body = len(masses)
    dim = 3

    F = np.zeros(2 * N_body * dim)

    # Vistas más claras
    Umat = U.reshape(N_body, 2 * dim)
    R = Umat[:, :dim]     # posiciones
    V = Umat[:, dim:]     # velocidades

    Fmat = F.reshape(N_body, 2 * dim)

    # dr/dt = v
    Fmat[:, :dim] = V

    # dv/dt = suma de fuerzas gravitatorias
    for i in range(N_body):
        acc = np.zeros(dim)
        for j in range(N_body):
            if i != j:
                diff = R[j] - R[i]
                dist = np.linalg.norm(diff)
                acc += masses[j] * diff / dist ** 3
        Fmat[i, dim:] = G * acc

    return F



#   Problema de Cauchy (Milestone 2)

def Cauchy_problem(F, U0, t, temporal_scheme):
    N = len(t) - 1
    m = len(U0)
    U = np.zeros((N + 1, m))
    U[0, :] = U0

    for n in range(N):
        U[n+1, :] = temporal_scheme(U[n, :], t[n], t[n+1], F)

    return U


#   PARÁMETROS DE SIMULACIÓN


# Tiempo
T = 10
N_T = 2000
t = np.linspace(0, T, N_T)

# Masas de los cuerpos
masses = np.array([1.0, 0.1, 0.1])

# Condiciones iniciales (r0,v0,r1,v1,r2,v2)
U0 = np.array([
    # cuerpo 0
    -1,   0.5, 0,      0,   0.3, 0.1,
    # cuerpo 1
     1.2,-0.4, 0.3,   -0.2, 0.1, 0,
    # cuerpo 2
     0,   1.5,-0.2,    0.1,-0.1, 0.2
])

# Definimos F(U,t) con masas fijadas
F = lambda U, tt: N_body_problem(U, tt, masses=masses, G=1.0)

# Resolver el sistema
U = Cauchy_problem(F, U0, t, RK4)

N_body = len(masses)
dim = 3



#   PLOT DE ÓRBITAS


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(N_body):
    x = U[:, 2 * i * dim]
    y = U[:, 2 * i * dim + 1]
    z = U[:, 2 * i * dim + 2]
    ax.plot(x, y, z, label=f"Cuerpo {i}")

ax.set_title("Órbitas del problema N-cuerpos")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()