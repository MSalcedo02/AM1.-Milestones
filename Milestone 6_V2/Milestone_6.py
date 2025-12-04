from Lagrange_points import L123, L4, L5
from Stability import estabilidad_Li
from Cauchy_problem import Cauchy_problem, CR3BP
from Temporal_schemes import RK45, Runge_Kutta, Euler, Crank_Nicolson, LeapFrog

import numpy as np
import matplotlib.pyplot as plt



#  FUNCIÓN: órbita simple alrededor de un punto Li

def orbita_alrededor_Li(Li, mu, T, amp, vel, t_points, scheme=RK45):

    """
    # amp: perturbación inicial en posición.
    # vel: perturbación inicial en velocidad.
    # scheme: método numérico para integrar (por defecto RK45).
    # Se devuelve t y la trayectoria U(t).

    """
    xL, yL = Li  #coordenadas de punto de Lagrange
    U0 = np.array([xL + amp, yL, 0, vel])  # Vector de estado inicial al que se le agrega una perturbación inicial
    t = np.linspace(0, T, t_points)   #vector de tiempos equidistante
    F = lambda U, t: CR3BP(U, t, mu)  #funcion a integrar
    U = Cauchy_problem(F, U0, t, scheme)  #resolución ecuación diferencial
    return t, U



# FUNCIÓN: órbitas con varios esquemas (Euler, RK4, RK45…)

def simulate_orbits_around_Li(Li, mu, amp, vel, T, t_points, schemes):
    xL, yL = Li
    U0 = np.array([xL + amp, yL, 0, vel])  # condición inicial
    t = np.linspace(0, T, t_points)
    F = lambda U, t: CR3BP(U, t, mu)

    results = {}  #almacena resultados de cada esquema 
    for name, scheme in schemes.items():
        print(f"Simulando con esquema {name} ...")
        U = Cauchy_problem(F, U0, t, scheme)
        results[name] = U

    return t, results


# FUNCIÓN: generar MUCHAS órbitas alrededor de un L_i

def muchas_orbitas_Li(Li, mu, T, epsilons, vel, t_points, scheme=RK45, N=16,
                      factor_vel=1.0, spread=1.0):
    """
    Genera múltiples órbitas alrededor de un punto L_i con separación
    radial, angular y en velocidad para visualizar mejor la dinámica.

    spread: multiplica la separación radial real
    factor_vel: añade perturbación en velocidad
    """
    xL, yL = Li
    t = np.linspace(0, T, t_points)
    F = lambda U, t: CR3BP(U, t, mu)

    angulos = np.linspace(0, 2 * np.pi, N, endpoint=False)  #Distribución de órbitas en N angulos diferentes
    trayectorias = []  #almacena las órbitas

    for eps in epsilons:
        for th in angulos:

            # calculo de desplazamiento radial 
            dx = spread * eps * np.cos(th)
            dy = spread * eps * np.sin(th)

            # Cañculo velocidad tangencial 
            dvx = -factor_vel * eps * np.sin(th)
            dvy =  factor_vel * eps * np.cos(th)

            U0 = np.array([
                xL + dx,
                yL + dy,
                dvx,
                vel + dvy
            ])

            U = Cauchy_problem(F, U0, t, scheme)
            trayectorias.append(U)

    return t, trayectorias



# SISTEMA TIERRA–LUNA

mu = 0.01215


#  1. Cálculo de L1–L5 
L1 = L123(mu, [0.7, 0])
L2 = L123(mu, [1.2, 0])
L3 = L123(mu, [-1.0, 0])
L4_ = L4(mu)
L5_ = L5(mu)

print("L1 =", L1)
print("L2 =", L2)
print("L3 =", L3)
print("L4 =", L4_)
print("L5 =", L5_)


# 2. Estabilidad 
print("\nAutovalores de L1:", estabilidad_Li(L1, mu))
print("Autovalores de L2:", estabilidad_Li(L2, mu))
print("Autovalores de L3:", estabilidad_Li(L3, mu))
print("Autovalores de L4:", estabilidad_Li(L4_, mu))
print("Autovalores de L5:", estabilidad_Li(L5_, mu))



# 3. Órbita pequeña alrededor de L1 (Prueba)

amp = 1e-4
U0 = np.array([L1[0] + amp, L1[1], 0, 0.01])

T = 20
t = np.linspace(0, T, 4000)

F = lambda U, t: CR3BP(U, t, mu)
U = Cauchy_problem(F, U0, t, RK45)

plt.figure(figsize=(7, 7))
plt.plot(U[:, 0], U[:, 1], 'b', lw=0.8, label="Órbita alrededor de L1")
plt.plot(L1[0], L1[1], 'ro', label="L1")
plt.plot(0, 0, 'ko', label="Centro sistema")
plt.axis('equal')
plt.xlabel("x"), plt.ylabel("y")
plt.title("Órbita pequeña alrededor de L1")
plt.legend()
plt.show()



# 4. ÓRBITAS ALREDEDOR DE LOS 5 PUNTOS LAGRANGE 

"""

Se generan órbitas pequeñas alrededor de cada punto.

Se grafican todas juntas para comparar posición relativa y estabilidad

"""

puntos = {"L1": L1, "L2": L2, "L3": L3, "L4": L4_, "L5": L5_}

orbitales = {}

for nombre, Li in puntos.items():
    print(f"\nSimulando órbita alrededor de {nombre}...")

    if nombre in ["L1", "L2", "L3"]:
        amp = 1e-4
        vel = 0.01
    else:
        amp = 1e-3
        vel = 0.02

    t, U = orbita_alrededor_Li(Li, mu, T=20, amp=amp, vel=vel,
                               t_points=4000, scheme=RK45)
    orbitales[nombre] = U

plt.figure(figsize=(9, 9))

for nombre, U in orbitales.items():
    plt.plot(U[:, 0], U[:, 1], lw=0.9, label=f"Órbita {nombre}")

for nombre, Li in puntos.items():
    plt.plot(Li[0], Li[1], 'o', markersize=7, label=f"{nombre}")

plt.plot(0, 0, 'ko', markersize=6)
plt.xlabel("x"), plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.title("Órbitas alrededor de los 5 puntos de Lagrange")
plt.show()



#  5. MUCHAS ÓRBITAS ALREDEDOR DE CADA L_i 

for nombre, Li in puntos.items():
    print(f"\nGenerando muchas órbitas alrededor de {nombre}...")

    if nombre in ["L1", "L2", "L3"]:
        epsilons = [1e-4]
        vel = 0.01
    else:
        epsilons = [5e-4, 1e-3, 2e-3]
        vel = 0.02

    t, trayectorias = muchas_orbitas_Li(Li, mu, T=20,
                                        epsilons=epsilons,
                                        vel=vel, t_points=3000,
                                        scheme=RK45, N=20)

    plt.figure(figsize=(8, 8))
    for U in trayectorias:
        plt.plot(U[:, 0], U[:, 1], lw=0.6)

    plt.plot(Li[0], Li[1], 'ro')
    plt.title(f"Muchas órbitas alrededor de {nombre}")
    plt.axis("equal")
    plt.show()



#     6. ÓRBITAS ALREDEDOR DE L1 CON VARIOS ESQUEMAS (gráficas separadas)


schemes = {
    "Euler": Euler,
    "Crank-Nicolson": Crank_Nicolson,
    "RK4": Runge_Kutta,
    "Leapfrog": LeapFrog,
    "RK45": RK45
}

t, results = simulate_orbits_around_Li(
    Li=L1, mu=mu,
    amp=1e-4, vel=0.01,
    T=20, t_points=4000,
    schemes=schemes
)

# Graficar cada esquema por separado
for name, U in results.items():
    plt.figure(figsize=(7,7))
    plt.plot(U[:,0], U[:,1], lw=1.0, label=f"Órbita ({name})")
    plt.plot(L1[0], L1[1], 'ro', label="L1")
    plt.plot(0,0,'ko', label="Centro sistema")
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Órbita alrededor de L1 usando {name}")
    plt.show()