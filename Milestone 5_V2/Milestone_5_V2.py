from numpy import zeros, linspace, log, array
import matplotlib.pyplot as plt
from Temporal_schemes import Runge_Kutta, Euler, Inverse_Euler, LeapFrog, Crank_Nicolson
from Cauchy_problem import Cauchy_problem, N_body_problem

#   PARÁMETROS DE SIMULACIÓN
# Tiempo
T = 10
N_T = 2000
t = linspace(0, T, N_T)

# Masas de los cuerpos
masses = array([1.0, 0.1, 0.1])

# Condiciones iniciales (r0,v0,r1,v1,r2,v2)
U0 = array([
    # cuerpo 0
    -1,   0.5, 0,      0,   0.3, 0.1,
    # cuerpo 1
     1.2,-0.4, 0.3,   -0.2, 0.1, 0,
    # cuerpo 2
     0,   1.5,-0.2,    0.1,-0.1, 0.2
])

# Definimos F(U,t) con masas fijadas
F = lambda U, tt: N_body_problem(U, tt, masses=masses, G=1.0)

# Resolver el sistema
U = Cauchy_problem(F, U0, t, Runge_Kutta)

N_body = len(masses)
dim = 3



#   PLOT DE ÓRBITAS


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(N_body):
    x = U[:, 2 * i * dim]
    y = U[:, 2 * i * dim + 1]
    z = U[:, 2 * i * dim + 2]
    ax.plot(x, y, z, label=f"Cuerpo {i}")

ax.set_title("Órbitas del problema N-cuerpos")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()