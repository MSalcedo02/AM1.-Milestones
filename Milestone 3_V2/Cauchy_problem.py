# PROBLEMA DE CAUCHY
from numpy import zeros, linspace, log


def Cauchy_problem(F, U0, t, temporal_scheme):
    N = len(t) - 1
    Nv = len (U0)
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
