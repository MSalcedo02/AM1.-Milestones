import numpy as np
from Tools import Newton

def grad_U_eff(X, mu):
    x, y = X
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)

    """
    Derivadas parciales del potencial efectivo
    Representa las fuerzas que actúan sobre el tercer cuerpo.
    Encuentra puntos de equilibrio
    """

    dUx = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3
    dUy = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3

    return np.array([dUx, dUy])

def L123(mu, guess):

    """
    Posiciones de los tres puntos de Lagrange L_1, L_2, L_3
    Funcion lambda anónima que solo depende de X para usar en un método numérico.
    Raíz punto de Lagrange gradiente del potencial cero.
    guess es la estimación inicial para el método numérico

    """

    f = lambda X: grad_U_eff(X, mu)
    return Newton(f, guess)

def L4(mu):

    """
    Posición punto de Lagrange L_4
    """
    return np.array([0.5 - mu,  np.sqrt(3) / 2])

def L5(mu):
    """
    Posición punto de Lagrange L_5
    """
    return np.array([0.5 - mu, -np.sqrt(3) / 2])