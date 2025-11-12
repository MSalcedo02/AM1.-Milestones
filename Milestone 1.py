from numpy import array, concatenate, zeros, linspace
from numpy.linalg import norm
import matplotlib.pyplot as plt

def F(U): 
    
    #Returns F(U)= (\\dot(r), -r/|r|^3) donde r es el vector posición
       
    r = U[0:2]
    rd = U[2:4]

    return concatenate ((rd, -r/norm(r)**3), axis=None)

N = 20000 #maximun order od Euler's method 
delta_t = 0.001


########################################################     MÉTODO DE EULER     ###################################################

U_euler = zeros((N+1, 4))

U_euler[0,:] = array([1,0,0,1])

for n in range(0,N):
    U_euler[n+1, :] = U_euler[n, :] + delta_t * F(U_euler[n,:])


########################################################     MÉTODO RUNGE KUTTA     ###############################################


U_RK = zeros((N+1, 4))

U_RK[0,:] = array([1,0,0,1])

for n in range(0,N):
    k1 = F(U_RK[n,:])
    k2 = F(U_RK[n,:] + delta_t/2 * k1)
    k3 = F(U_RK[n,:] + delta_t/2 * k2)
    k4 = F(U_RK[n,:]+ delta_t * k3)

    U_RK[n+1, :] = U_RK[n, :] + delta_t/6 * (k1 + 2 * k2 + 2 * k3 + k4)


########################################################     MÉTODO CRANK-NICHOLSON     ###########################################


U_CN = zeros((N+1, 4))

U_CN[0,:] = array([1,0,0,1])  #posocion (x0, y0, vx, vy)

#iteracion de punto fijo 
iter_max = int(200)

for n in range(0,N):
    U_n = U_CN[n, :]
    F_n = F(U_n)

    U_0 = U_n + delta_t * F_n

    for i in range (iter_max): 

        F_CN_aux = F(U_0)
        U_CN_aux = U_n + 0.5 * delta_t * (F_n + F_CN_aux)

        if norm(U_CN_aux - U_0) < 1e-10:
            break  

        U_0 = U_CN_aux

    U_CN[n+1, :] = U_CN_aux


########################################################     GRAFICACIÓN DE MÉTODOS     ###########################################


plt.axis("equal")
plt.plot(U_euler[:, 0], U_euler[:, 1], label="Euler", color = 'red')
plt.plot(U_RK[:, 0], U_RK[:, 1], label="Runge kutta", color = 'blue')
plt.plot(U_CN[:, 0], U_CN[:, 1], label="Crank nicolson", color = 'green')
plt.legend()
plt.title("Comparación de métodos-órbita kepleriana")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()