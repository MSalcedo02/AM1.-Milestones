from Cauchy_problem import Cauchy_error
from numpy import zeros, log10, polyfit, sqrt, abs, maximum, linspace
from numpy.linalg import norm

def refine_mesh(t1):
    """
    Ejemplo: t1 = [0, 1, 2] --> t2 = [0, 0.5, 1, 1.5, 2]
    """
    N = len(t1) - 1  

    t2 = zeros(2*N+1) 
    for i in range(0, N): 
        t2[2*i] = t1[i]
        t2[2*i+1] = (t1[i] + t1[i+1])/2 
    
    t2[2*N] = t1[N]      

    return t2


def convergence_rate(Temporal_scheme, F, U0, t):
    
    N_meshes = 8          

    logN = zeros(N_meshes)
    logE = zeros(N_meshes) 

    t_i = t
    for i in range(N_meshes):
        N = len(t_i) - 1
        U, E = Cauchy_error(F, U0, t_i, Temporal_scheme) # Cualquier q, por defecto q = 1

        logN[i] = log10(N)
        logE[i] = log10(norm(E[N, :])) # Norma del punto con mas error (el ultimo)
        
        # Se refina la malla para la siguiente iteracion
        t_i = refine_mesh(t_i)       

    y = logE[logE > -12]
    x = logN[0:len(y)]
    m, b = polyfit(x, y, 1)    
    q = -m

    # Se recalcula logE una vez obtenido el orden
    logE = logE - log10(1 - 1/2**abs(q))   

    return logN, logE, q


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
    r1 = w + sqrt(w**2 + 1)
    r2 = w - sqrt(w**2 + 1)
    return maximum(abs(r1), abs(r2))



#  Regiones de estabilidad


def region_estabilidad(func, N=400, x0=-3, xf=3, y0=-3, yf=3):
    X = linspace(x0, xf, N)
    Y = linspace(y0, yf, N)
    W = X + 1j*Y[:, None]

    R = func(W)
    return abs(R) <= 1