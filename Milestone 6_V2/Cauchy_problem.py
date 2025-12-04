# PROBLEMA DE CAUCHY
from numpy import zeros, linspace, array
from numpy.linalg import norm 


def Cauchy_problem(F, U0, t, temporal_scheme):
    N = len(t) - 1
    Nv = len (U0)  #Numero de variables
    U = zeros((N + 1, Nv))
    U[0, :] = U0
    for n in range(N):
        U[n + 1, :] = temporal_scheme(U[n, :], t[n], t[n + 1], F)
    return U 


# ESTIMACIÓN DE ERROR (RICHARDSON)


def Cauchy_error(F, U0, t, temporal_scheme, q=1):
    N = len(t) - 1
    t_fine = linspace(t[0], t[-1], 2*N + 1)

    # Soluciones gruesa y fina
    U_h = Cauchy_problem(F, U0, t, temporal_scheme)
    U_2h = Cauchy_problem(F, U0, t_fine, temporal_scheme)

    # Richardson
    E = (U_2h[::2] - U_h) / (1 - 2**(-q))

    return U_h, E


#   PROBLEMA N-CUERPOS


def N_body_problem(U, t, masses, G=1.0):

    N_body = len(masses)
    dim = 3

    F = zeros(2 * N_body * dim)

    # Vistas más claras
    Umat = U.reshape(N_body, 2 * dim)
    R = Umat[:, :dim]     # posiciones
    V = Umat[:, dim:]     # velocidades

    Fmat = F.reshape(N_body, 2 * dim)

    # dr/dt = v
    Fmat[:, :dim] = V

    # dv/dt = suma de fuerzas gravitatorias
    for i in range(N_body):
        acc = zeros(dim)
        for j in range(N_body):
            if i != j:
                diff = R[j] - R[i]
                dist = norm(diff)
                acc += masses[j] * diff / dist ** 3
        Fmat[i, dim:] = G * acc

    return F


#   Problema de 3 cuerpos circular 

def CR3BP(U, t, mu):
    """
    r1 es la distancia al cuerpo 1
    r2 es la distancia al cuerpo 2
    
    """
    x, y, vx, vy = U

    r1 = ((x + mu) ** 2 + y ** 2) ** 0.5
    r2 = ((x - (1 - mu)) ** 2 + y ** 2) ** 0.5

    """

    Aceleraciones en x e y del tercer cuerpo usando las ecuaciones del CR3BP
    2*vy y -2*vx términos de Coriolis
    r1**3 y r2**3 atracción gravitatoria normalizada

    """


    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3
    ay = - 2 * vx + y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3

    return array([vx, vy, ax, ay])