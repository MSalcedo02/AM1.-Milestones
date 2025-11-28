from numpy.linalg import norm
from numpy import concatenate


def kepler(U, t):
    r = U[0:2]
    rd = U[2:4]
    return concatenate((rd, - r / norm(r)**3))