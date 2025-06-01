from numba import jit, njit
import utils

@njit
def energy_potts(x, Jij):
    e = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            e -= Jij[i, j, x[i], x[j]]
    return e


@njit
def energy_nskew(x, h, J, J2):
    counts = utils.aacounts_int_jit(x)
    q = len(h)
    e = 0
    for alpha in range(q):
        e -= h[alpha]*counts[alpha]
        for beta in range(alpha, q):
            e -= J[alpha, beta]*counts[alpha]*counts[beta]
            for gamma in range(beta, q):
                e -= J2[alpha, beta, gamma]*counts[alpha]*counts[beta]*counts[gamma]
    return e

@njit
def energy_full_model(x, h, J, J2, Jij):
    return energy_potts(x, Jij) + energy_nskew(x, h, J, J2)