######################################                          CAUCHY_ERROR                          ################################################

# Argumentos de entrada: (F, u0, t, temporal_scheme)
# Argumentos de salida: (U (solución problema de Cauchy), Error)

###################################################################################################################################################### 
from numpy import zeros, linspace, array, concatenate, log
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt



# SISTEMA DIFERENCIAL (PROBLEMA DE 2 CUERPOS)


def F(U, t):
    r = U[0:2]
    rd = U[2:4]
    return concatenate((rd, - r / norm(r)**3))


# NEWTON — JACOBIANO — GAUSS


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


# METODOS TEMPORALES


def Euler(U1, t1, t2, F):
    dt = t2 - t1 
    return U1 + dt * F(U1, t1)


def Inverse_Euler(U1, t1, t2, F):
    dt = t2 - t1
    def G(x):
        return x - U1 - dt * F(x, t2)
    return Newton(G, U1)


def Crank_Nicolson(U1, t1, t2, F):
    dt = t2 - t1 
    a = U1 + dt / 2 * F(U1, t1)
    def G(x):
        return x - a - dt/2 * F(x, t2)
    return Newton(G, U1)


def RK4(U1, t1, t2, F):
    dt = t2 - t1
    k1 = F(U1, t1)
    k2 = F(U1 + 0.5 * dt * k1, t1 + 0.5 * dt)
    k3 = F(U1 + 0.5 * dt * k2, t1 + 0.5 * dt)
    k4 = F(U1 + dt * k3, t2)
    return U1 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# PROBLEMA DE CAUCHY


def Cauchy_problem(F, U0, t, temporal_scheme):
    N = len(t) - 1
    Nv = len (U0)
    U = zeros((N + 1, Nv))
    U[0, :] = U0
    for n in range(N):
        U[n + 1, :] = temporal_scheme(U[n, :], t[n], t[n + 1], F)
    return U 


# ESTIMACIÓN DE ERROR (RICHARDSON)


def Cauchy_error(F, U0, t, temporal_scheme, q):
    N = len(t) - 1
    t_fine = linspace(t[0], t[-1], 2*N + 1)

    # Soluciones gruesa y fina
    U_h = Cauchy_problem(F, U0, t, temporal_scheme)
    U_2h = Cauchy_problem(F, U0, t_fine, temporal_scheme)

    # Richardson
    E = (U_2h[::2] - U_h) / (1 - 2**(-q))

    return U_h, E


# CALCULO DE ORDEN DE CONVERGENCIA


def convergence_rate(errors, dts):
    p = zeros(len(errors) - 1)
    for i in range(len(errors) - 1):
        p[i] = log(errors[i] / errors[i + 1]) / log(dts[i] / dts[i + 1])
    return p


# TESTS Y GRAFICAS


def test_Cauchy():
#Variableeees:
    U0 = ([1, 0, 0, 1])
    T = 40
    N = 2000
    t = linspace (0, T, N + 1)

    U = Cauchy_problem(F, U0, t, Euler)

#trayectoria
    plt.plot(U[:, 0], U[:, 1], color = 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trayectoria en el plano")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def test_error():
    U0 = ([1, 0, 0, 1])
    T = 40
    N = 2000
    t = linspace (0, T, N + 1)
    q = 1 
    U, E = Cauchy_error (F, U0, t, Euler, q)
    plt.plot(t, E[:, 0], color = 'blue')
    plt.xlabel("t")
    plt.ylabel("Error en x")
    plt.title("Error por Richardson")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def test_convergence():
    U0 = array([1, 0, 0, 1])
    T = 40
    Ns = [500, 1000, 2000, 4000]

    methods = [Euler, Inverse_Euler, Crank_Nicolson, RK4]
    names = ["Euler", "Inverse Euler", "Crank-Nicolson", "RK4"]
    q_values = [1, 1, 2, 4]   # órdenes teóricos

    plt.figure()

    for method, name, q in zip(methods, names, q_values):
        errors = []
        dts = []

        for N in Ns:
            t = linspace(0, T, N + 1)
            U, E = Cauchy_error(F, U0, t, method, q)
            errors.append(norm(E[-1]))     # error en el tiempo final
            dts.append(T / N)

        p = convergence_rate(errors, dts)
        print(f"\nMétodo {name}: orden aproximado = {p}")

        plt.loglog(dts, errors, '-o', label=name)

    plt.xlabel("dt")
    plt.ylabel("Error final")
    plt.title("Convergencia de métodos temporales")
    plt.legend()
    plt.grid(True)
    plt.show()


# LLAMADA A TESTS 


# test_Cauchy()
# test_error()
test_convergence()


