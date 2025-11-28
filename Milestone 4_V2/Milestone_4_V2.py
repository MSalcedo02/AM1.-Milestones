import Cauchy_problem
from  Convergence_and_estability import R_CN, R_Euler, R_EI, R_LF, R_RK4, region_estabilidad
from Differential_equation import simular
from Temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, Runge_Kutta, LeapFrog
import Tools
import numpy as np
import matplotlib.pyplot as plt


#  Simulación del oscilador


y0 = np.array([1, 0])
h = 0.1
T = 20

metodos = {
    "Euler": Euler,
    "Inverse Euler": Inverse_Euler,
    "Crank–Nicolson": Crank_Nicolson,
    "RK4": Runge_Kutta,
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

cmap = "turbo"   # puedes cambiar: "plasma", "inferno", "magma", "turbo"

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