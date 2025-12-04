import numpy as np

def Hessian_Ueff(X, mu):

    """
    X = [x, y]: posición del tercer cuerpo.
    
    """
    x, y = X

    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)

    """
    segunda derivada del potencial respecto a x
    """
    Uxx = 1 - (1 - mu) * (1 / r1 ** 3 - 3 * (x+mu) ** 2 / r1 ** 5) - mu * (1 / r2 ** 3 - 3 * (x - (1 - mu)) ** 2 / r2 ** 5)
    """
    segunda derivada del potencial respecto a y
    """
    Uyy = 1 - (1 - mu) * (1 / r1 ** 3 - 3 * y ** 2 / r1 ** 5) - mu * (1 / r2 ** 3 - 3 * y ** 2 / r2 ** 5)

    """
    Derivada cruzada
    """
    Uxy = 3 * y * ((1 - mu) * (x + mu) / r1 ** 5 + mu * (x - (1 - mu)) / r2 ** 5)

    return Uxx, Uyy, Uxy

def Jacobiano_Lagrange(X, mu):
    """
    matriz jacobiana del sistema linealizado alrededor de un punto de Lagrange
    
    """
    Uxx, Uyy, Uxy = Hessian_Ueff(X, mu)

    """
    primeras dos filas posición y velocidad
    últimas dos filas Hessiano y los términos de Coriolis
    """

    A = np.array([
        [0,   0,   1,   0],
        [0,   0,   0,   1],
        [Uxx, Uxy, 0,   2],
        [Uxy, Uyy, - 2,  0]
    ])
    return A

def estabilidad_Li(X, mu):
    """
    estabilidad lineal de un punto de Lagrange L_i
    matriz jacobiana en el punto X
    """
    A = Jacobiano_Lagrange(X, mu)

    """
    valores propios de la matriz jacobiana
    """

    eigvals = np.linalg.eigvals(A)
    return eigvals