import numpy as np
from numba import jit, njit
import utils

from compute_energies import energy_potts, energy_nskew, energy_full_model


def fit_full_model(train_matrix, sampler, h=None, J=None, J2=None, q=utils.naminoacids,
            niter=1, epsilon=0.1, pseudocount=1.0,
            prng=None, output=False):
    """ sampler(x0, energy, jump, prng=prng): function returning samples from the distribution """

    if prng is None:
        prng = np.random

    # calculate empirical observables
    _, L = train_matrix.shape
    aacounts = utils.to_aacounts(train_matrix)
    n1 = calc_n1(aacounts)
    n2 = calc_n2(aacounts)
    n3 = calc_n3(aacounts)

    fi = utils.frequencies(train_matrix, num_symbols=q, pseudocount=pseudocount)
    fij = utils.pair_frequencies(train_matrix, num_symbols=q, fi=fi, pseudocount=pseudocount)


    # initialize Lagrange multipliers
    if h is None:
        h = np.log(n1/L)
        h -= np.mean(h)
    else:
        h = h.copy()
    if J is None:
        J = np.zeros_like(n2)
    else:
        J = J.copy()
    if J2 is None:
        J2 = np.zeros_like(n3)
    else:
        J2 = J2.copy()
    hi = np.zeros_like(fi)
    Jij = np.zeros_like(fij)
   
    for iteration in range(niter):
        if output:
            print('iteration %g/%g'%(iteration+1,niter))

        x0 = global_jump(np.zeros(L), q, prng=prng)
        
        @njit
        def jump(x):
            return local_jump_jit(x, q)
        @njit
        def energy(x):
            return energy_full_model(x, h, J, J2, Jij)

        samples = sampler(x0, energy, jump)
        aacounts = utils.to_aacounts(samples)

        n1_model = calc_n1(aacounts)
        n2_model = calc_n2(aacounts)
        n3_model = calc_n3(aacounts)

        fi_model = utils.frequencies(samples, q, pseudocount=pseudocount)
        fij_model = utils.pair_frequencies(samples, q, fi_model, pseudocount=pseudocount)
 
        h -= np.log(n1_model/n1)*epsilon
        J -= np.log(n2_model/n2)*epsilon
        J2 -= np.log(n3_model/n3)*epsilon

        Jij -= np.log(fij_model/fij)*epsilon

    return h, J, J2, Jij, hi



def calc_n1(aacounts):
    return np.mean(aacounts, axis=0)


@njit
def calc_n2(matrix):
    N, q = matrix.shape
    n2 = np.zeros((q, q))
    for s in range(N):
        for alpha in range(q):
            for beta in range(q):
                n2[alpha, beta] += matrix[s, alpha]*matrix[s, beta]     
    n2 /= N
    return n2

@njit
def calc_n3(matrix):
    N, q = matrix.shape
    n3 = np.zeros((q, q, q))
    for s in range(N):
        for alpha in range(q):
            for beta in range(q):
                for gamma in range(q):
                    n3[alpha, beta, gamma] += matrix[s, alpha]*matrix[s, beta]*matrix[s, gamma]
    n3 /= N
    return n3


def global_jump(x, q, prng=None):
    if prng is None:
        prng = np.random
    return prng.randint(q, size=len(x))


@jit(nopython=True)
def local_jump_jit(x, q, seed=None):
    prng = np.random
    if not (seed is None):
        prng.seed(seed)
    xnew = x.copy()
    index = prng.randint(len(x))
    xnew[index] = (x[index] + prng.randint(1, q))%q
    return xnew


if __name__ == "__main__":
    matrix = utils.load_matrix('Listeria_kmer_matrix_train.csv.gz')
    L = 9
    nsample = L
    output = True
    q = utils.naminoacids
    pseudocount = 1.0

    # the parameters that are commented out are more realistic, but smaller values are chosen for efficiency in running
    # the tutorial code

    # niter = 200
    niter = 50
    stepsize = 0.01 
    # nsteps = 1e7
    nsteps = 1e5
    # nburnin = 1e3
    nburnin = 10

    prng = np.random
    
    def sampler(*args, **kwargs):
        return utils.mcmcsampler(*args, nsteps=nsteps, nsample=nsample, nburnin=nburnin)


    h, J, J2, Jij, hi = fit_full_model(matrix, sampler=sampler, h=None, J=None, J2=None,
                    niter=niter, pseudocount=pseudocount,
                    epsilon=stepsize, prng=prng, output=output)

    np.savez('Listeria_model_params.npz', h=h, J=J, J2=J2, Jij=Jij, hi=hi)