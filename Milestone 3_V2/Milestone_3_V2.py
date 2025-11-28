from Temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, Runge_Kutta
from Cauchy_problem import Cauchy_problem, Cauchy_error
from Differential_equation import kepler
from Convergence_and_estability import convergence_rate

from numpy import zeros, linspace, pi
import matplotlib.pyplot as plt


def test_error():
    U0 = ([1, 0, 0, 1])
    T = 40
    N = 2000
    t = linspace (0, T, N + 1)
    q = 1 
    U, E = Cauchy_error (kepler, U0, t, Runge_Kutta, q)
    plt.plot(t, E[:, 0], color = 'blue')
    plt.xlabel("t")
    plt.ylabel("Error en x")
    plt.title("Error por Richardson")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def test_convergence():
    U0 = ([1, 0, 0, 1])
    T = 8*pi
    N = 1000
    t = linspace(0, T, N+1)

    logN, logE, q = convergence_rate(Runge_Kutta, kepler, U0, t)

    print(f"The order of the temporal scheme is: {q}")    

    # Plot trajectory
    plt.plot(logN, logE)
    plt.xlabel('logN')
    plt.ylabel('logE')
    plt.title('Order of the temporal scheme')
    plt.axis('equal')
    plt.grid(True)
    plt.show() 


# LLAMADA A TESTS (Descomentar la que quieras)

# test_error()
test_convergence()