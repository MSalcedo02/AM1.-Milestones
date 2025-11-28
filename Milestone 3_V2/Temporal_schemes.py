from Tools import Newton

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