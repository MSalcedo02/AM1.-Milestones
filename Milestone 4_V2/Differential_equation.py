from numpy.linalg import norm
from numpy import concatenate, array, linspace, zeros 


def kepler(U, t):
    r = U[0:2]
    rd = U[2:4]
    return concatenate((rd, - r / norm(r)**3))

#  Oscilador lineal


def Oscillator(y, t):
    # y = [x, v],  x' = v,  v' = -x
    return array([y[1], -y[0]])


def simular(metodo, y0, h, T):
    N = int(T/h)
    t = linspace(0, T, N)
    y = zeros((N, len(y0)))
    y[0] = y0

    for n in range(N-1):
        y[n+1] = metodo(y[n], t[n], t[n+1], Oscillator)

    return t, y
