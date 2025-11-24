import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve



#  Funciones: Derivada, Jacobiano, Newton


def derivada(f, x, dx):
    h = 1e-7
    return (f(x + dx) - f(x - dx)) / (2 * h)

def Jacobiano(f, x):
    J = np.zeros((len(x), len(x)))
    for j in range(len(x)):
        dx = np.zeros(len(x))
        dx[j] = 1e-7
        J[:, j] = derivada(f, x, dx)
    return J

def Gauss(A, b):
    return solve(A, b)

def Newton(f, x0):
    x = x0.copy()
    Dx = 1.0
    while norm(Dx) > 1e-10:
        A = Jacobiano(f, x)
        Dx = Gauss(A, -f(x))
        x = x + Dx
    return x



#  Métodos numéricos temporales


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
    a = U1 + dt/2 * F(U1, t1)
    def G(x):
        return x - a - dt/2 * F(x, t2)
    return Newton(G, U1)

def RK4(U1, t1, t2, F):
    dt = t2 - t1
    k1 = F(U1, t1)
    k2 = F(U1 + 0.5*dt*k1, t1 + 0.5*dt)
    k3 = F(U1 + 0.5*dt*k2, t1 + 0.5*dt)
    k4 = F(U1 + dt*k3, t2)
    return U1 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def LeapFrog(U1, t1, t2, F):
    dt = t2 - t1
    # U = [x, v]
    x, v = U1

    # Primer medio paso para v (kick)
    v_half = v + 0.5 * dt * (-x)

    # Paso completo de x
    x_new = x + dt * v_half

    # Kick final
    v_new = v_half + 0.5 * dt * (-x_new)

    return np.array([x_new, v_new])



#  Oscilador lineal


def F(y, t):
    # y = [x, v],  x' = v,  v' = -x
    return np.array([y[1], -y[0]])


def simular(metodo, y0, h, T):
    N = int(T/h)
    t = np.linspace(0, T, N)
    y = np.zeros((N, len(y0)))
    y[0] = y0

    for n in range(N-1):
        y[n+1] = metodo(y[n], t[n], t[n+1], F)

    return t, y



#  Funciones de estabilidad


def R_Euler(w):
    return 1 + w

def R_RK4(w):
    return 1 + w + w**2/2 + w**3/6 + w**4/24

def R_CN(w):
    return (1 + w/2) / (1 - w/2)

def R_EI(w):
    return 1 / (1 - w)

def R_LF(w):
    r1 = w + np.sqrt(w**2 + 1)
    r2 = w - np.sqrt(w**2 + 1)
    return np.maximum(np.abs(r1), np.abs(r2))



#  Regiones de estabilidad


def region_estabilidad(func, N=400, x0=-3, xf=3, y0=-3, yf=3):
    X = np.linspace(x0, xf, N)
    Y = np.linspace(y0, yf, N)
    W = X + 1j*Y[:, None]

    R = func(W)
    return np.abs(R) <= 1



#  Simulación del oscilador


y0 = np.array([1, 0])
h = 0.1
T = 20

metodos = {
    "Euler": Euler,
    "Inverse Euler": Inverse_Euler,
    "Crank–Nicolson": Crank_Nicolson,
    "RK4": RK4,
    "Leap-Frog": LeapFrog
}

soluciones = {}
for nombre, metodo in metodos.items():
    t, y = simular(metodo, y0, h, T)
    soluciones[nombre] = (t, y)



#  Graficar oscilador


plt.figure(figsize=(10,6))
for nombre, (t, y) in soluciones.items():
    plt.plot(t, y[:,0], label=nombre)

plt.title("Oscilador lineal — Comparación de métodos")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True)
plt.legend()
plt.show()



#  Graficar regiones de estabilidad



regiones = {
    "Euler": lambda w: R_Euler(w),
    "RK4": lambda w: R_RK4(w),
    "Crank–Nicolson": lambda w: R_CN(w),
    "Inverse Euler": lambda w: R_EI(w),
    "Leap-Frog": lambda w: R_LF(w)
}

cmap = "turbo"   

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

i = 0
for nombre, func in regiones.items():
    reg = region_estabilidad(func)

    ax = axs[i]
    ax.imshow(reg, extent=[-3,3,-3,3], origin="lower", cmap=cmap)

    ax.set_title(f"Región de estabilidad — {nombre}")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")

    # Ejes
    ax.axhline(0, color="white", lw=1)
    ax.axvline(0, color="white", lw=1)

    # Cuadrícula
    ax.grid(color="white", alpha=0.25)

    i += 1

# Dejar el último panel vacío
axs[-1].axis("off")

plt.tight_layout()

plt.show()
