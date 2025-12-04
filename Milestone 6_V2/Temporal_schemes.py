from Tools import Newton
from numpy import array
import numpy as np

def Euler(U1, t1, t2, F):
    dt = t2 - t1 
    return U1 + dt * F(U1, t1)

def Inverse_Euler(U, t1, t2, F):
    dt = t2 - t1
    def G(x):
        return x - U - dt * F(x, t2)
    return Newton(G, U)

def Crank_Nicolson(U1, t1, t2, F):
    dt = t2 - t1 
    a = U1 + dt / 2 * F(U1, t1)
    def G(x):
        return x - a - dt/2 * F(x, t2)
    return Newton(G, U1)

def Runge_Kutta(U1, t1, t2, F):
    dt = t2 - t1
    k1 = F(U1, t1)
    k2 = F(U1 + dt/2 * k1, t1 + dt/2)
    k3 = F(U1 + dt/2 * k2, t1 + dt/2)
    k4 = F(U1 + dt * k3, t1 + dt)
    return U1 + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def LeapFrog(U1, t1, t2, F):
    dt = t2 - t1

    # Variables
    x, y, vx, vy = U1

    # Aceleración inicial
    ax1, ay1 = F(U1, t1)[2:]

    # Kick - medio paso en la velocidad
    vx_half = vx + 0.5 * dt * ax1
    vy_half = vy + 0.5 * dt * ay1

    # Drift - paso completo en posición
    x_new = x + dt * vx_half
    y_new = y + dt * vy_half

    # Aceleración en nueva posición
    U_temp = np.array([x_new, y_new, vx_half, vy_half])
    ax2, ay2 = F(U_temp, t1 + dt)[2:]

    # Kick final
    vx_new = vx_half + 0.5 * dt * ax2
    vy_new = vy_half + 0.5 * dt * ay2

    return np.array([x_new, y_new, vx_new, vy_new])

#   MÉTODO EMBEBIDO RK45 (Dormand–Prince)

def RK45(U1, t1, t2, F):
    dt = t2 - t1

    # Coeficientes DP(4,5)
    """
    Coeficientes de tiempo intermedio para cada subpaso de RK.
    """

    a = [0,
         1/5,
         3/10,
         4/5,
         8/9,
         1,
         1]

    """
    Coeficientes para combinar las derivadas intermedias k[j] en cada subpaso.
    """

    b = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, - 56 / 15, 32 /9 ],
        [19372 / 6561, - 25360 / 2187, 64448 / 6561, - 212 / 729],
        [9017 / 3168, - 355 / 33, 46732 / 5247, 49 / 176, - 5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, - 2187 / 6784, 11 / 84]
    ]

    """
    c4: pesos para solución de orden 5
    c5: pesos para solución de orden 4
    """

    c4 = np.array([35 / 384, 0, 500 / 1113, 125 / 192,
                   -2187 / 6784, 11 / 84, 0])  # 5th order
    c5 = np.array([5179 / 57600, 0, 7571 / 16695,
                   393 / 640, - 92097 / 339200,
                   187 / 2100,  1 /40])         # 4th order

    k = []

    """
    arg comienza con el estado inicial U1.
    Se suman las contribuciones de k[j] ponderadas por b[i][j] para cada subpaso.
    Se evalúa la derivada F(arg, t1 + a[i]*dt) y se guarda en k[i].
    """
    for i in range(7):
        arg = U1.copy()
        for j in range(i):
            arg = arg + dt * b[i][j] * k[j]
        k.append(F(arg, t1 + a[i] * dt))

    # Alta y baja orden
    """
    y5: solución de orden 5 (más precisa).
    y4: solución de orden 4 (menos precisa).
    """
    y5 = U1 + dt * sum(c4[j] * k[j] for j in range(7))
    y4 = U1 + dt * sum(c5[j] * k[j] for j in range(7))

    # Error estimado
    error = np.linalg.norm(y5 - y4)

    return y5   # devolver la solución de orden mayor