from numpy import zeros, linspace, array, concatenate
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

# ---------------- Funciones existentes ----------------
def F(U, t):
    r = U[0:2]
    rd = U[2:4]
    return concatenate((rd, - r / norm(r) ** 3), axis=None)

def derivada(f, x, dx):
    h = 1e-7
    return (f(x + dx) - f(x - dx)) / (2 * h)

def Jacobiano(f, x):
    J = zeros((len(x), len(x)))
    for j in range(len(x)):
        dx = zeros(len(x))
        dx[j] = 1e-7
        J[:, j] = derivada(f, x, dx)
    return J 

def Gauss(A, b):
    return solve(A, b)

def Newton(f, x0):
    x = x0
    Dx = 1.0
    while norm(Dx) > 1e-10:
        A = Jacobiano(f, x)
        Dx = Gauss(A, -f(x))
        x = x + Dx
    return x

def Euler(U1, t1, t2, F):
    dt = t2 - t1 
    return U1 + dt * F(U1, t1)

def Crank_Nicolson(U1, t1, t2, F):
    dt = t2 - t1 
    a = U1 + dt / 2 * F(U1, t1)
    def G(x):
        return x - a - dt/2 * F(x, t2)
    return Newton(G, U1)

# ---------------- Nueva función Runge-Kutta 4 ----------------
def Runge_Kutta(U1, t1, t2, F):
    dt = t2 - t1
    k1 = F(U1, t1)
    k2 = F(U1 + dt/2 * k1, t1 + dt/2)
    k3 = F(U1 + dt/2 * k2, t1 + dt/2)
    k4 = F(U1 + dt * k3, t1 + dt)
    return U1 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# ---------------- Función genérica para resolver el problema ----------------
def Cauchy_problem(F, U0, t, temporal_scheme):
    N = len(t) - 1
    Nv = len(U0)
    U = zeros((N + 1, Nv))
    U[0, :] = U0
    for n in range(N):
        U[n + 1, :] = temporal_scheme(U[n, :], t[n], t[n + 1], F)
    return U 

# ---------------- Variables ----------------
U0 = array([1, 0, 0, 1])
T = 200
N = 10000
t = linspace(0, T, N + 1)

# ---------------- Soluciones ----------------
U_Euler = Cauchy_problem(F, U0, t, Euler)
U_CN = Cauchy_problem(F, U0, t, Crank_Nicolson)
U_RK = Cauchy_problem(F, U0, t, Runge_Kutta)

# ---------------- Gráfico comparativo ----------------
plt.plot(U_Euler[:,0], U_Euler[:,1], color='red', label='Euler')
plt.plot(U_CN[:,0], U_CN[:,1], color='green', label='Crank-Nicolson')
plt.plot(U_RK[:,0], U_RK[:,1], color='blue', label='Runge-Kutta 4')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trayectorias en el plano")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()
