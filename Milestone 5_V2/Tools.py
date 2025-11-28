from numpy import zeros
from numpy.linalg import norm, solve

#####################################################   HERRAMIENTAS   #################################################3

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