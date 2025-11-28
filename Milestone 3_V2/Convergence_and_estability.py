from Cauchy_problem import Cauchy_error
from numpy import zeros, log10, polyfit
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
