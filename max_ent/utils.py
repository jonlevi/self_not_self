from numba import jit, njit
import random
from functools import partial
from collections import defaultdict
import numpy as np
import pandas as pd

from mimetypes import guess_type
from functools import partial
import os.path
from itertools import groupby

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
aminoacids_set = set(aminoacids)
naminoacids = len(aminoacids)

_aatonumber = {c: i for i, c in enumerate(aminoacids)}
_numbertoaa = {i: c for i, c in enumerate(aminoacids)}

def map_aatonumber(seq):
    """
    Map sequence to array of number
    """
    seq = np.array(list(seq))
    return np.vectorize(_aatonumber.__getitem__)(seq)

def map_numbertoaa(seq):
    """
    Map integer to amino acid sequence
    """
    seq = list(seq)
    return np.vectorize(_numbertoaa.__getitem__)(seq)


def aatonumber(char):
    return _aatonumber[char]


def map_matrix(matrix, map_=_aatonumber):
    """
    Remap elements in a numpy array 

    Parameters
    ----------
    array : np.array
        Matrix to be remapped
    map_ : dict
        Map to be applied to matrix elements

    Returns
    -------
    np.array
        Remapped matrix
    """
    return np.vectorize(map_.__getitem__)(matrix)

def kmers_to_matrix(kmers):
    """"
    Map a list of str kmers to an integer numpy array.

    Parameters
    ----------
    kmers : iterable of strings
        kmers to be converted
    Returns
    -------
    np.array
        Mapped array
    """
    matrix_str =  np.array([list(kmer) for kmer in kmers])
    matrix = map_matrix(matrix_str)
    return matrix


def matrix_to_kmers(matrix):
    """"
    Map an integer numpy array to a list of str kmers.

    Parameters
    ----------
    matrix: np.array
        Array to be converted
    Returns
    -------
    iterable of strings
        kmers
    """
    char_matrix = map_numbertoaa(matrix)
    kmers = [''.join(row) for row in char_matrix]
    return kmers


@jit
def mcmcsampler(x0, energy, jump, nsteps=1000, nburnin=0, nsample=1):
    """Markov chain Monte carlo sampler (JIT enabled).

    x0: starting position (array)
    energy(x): function for calculating energy
    jump(x): function for calculating a proposed new position
    nburnin: burnin period in which states are not saved
    nsample: sample interval for saving states
    
    returns array of states
    """
    prng = np.random
    nsteps, nburnin, nsample = int(nsteps), int(nburnin), int(nsample)
    x = x0
    Ex = energy(x)
    samples = np.zeros(((nsteps-nburnin)//nsample, x0.shape[0]), dtype=np.int64)
    counter = 0
    for i in range(1, nsteps+1):
        xp = jump(x)
        Exp = energy(xp)
        if (Exp < Ex) or (prng.rand() < np.exp(-Exp+Ex)):
            x = xp
            Ex = Exp
        if (i > nburnin) and ((i-nburnin) % nsample == 0):
            samples[counter] = x
            counter += 1
    return samples


codon_map = {"UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
"UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
"UAU":"Y", "UAC":"Y", "UAA":"STOP", "UAG":"STOP",
"UGU":"C", "UGC":"C", "UGA":"STOP", "UGG":"W",
"CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
"CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
"CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
"CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
"AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
"ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
"AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
"AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
"GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
"GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",
"GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
"GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G",}


nt_to_ind = {
    'A' : 0,
    'C' : 1,
    'G' : 2,
    'U' : 3
    }
def ntfreq_to_aafreq(ntfreq):
    frequencies = {aa:0 for aa in aminoacids}
    for nts, aa in codon_map.items():
        if not aa == 'STOP':
            frequencies[aa] += np.prod([ntfreq[nt_to_ind[nt]] for nt in nts])
    sum_ = sum(frequencies.values())
    for aa in aminoacids:
        frequencies[aa] /= sum_
    return frequencies



def fasta_iter(fasta_name, returnheader=True, returndescription=False):
    """
    Given a fasta file return a iterator over tuples of header, complete sequence.
    """
    if returnheader and returndescription:
        raise Exception('one of returnheader/returndescription needs to be False')
    if guess_type(fasta_name)[1] =='gzip':
        _open = partial(gzip.open, mode='rt')
    else:
        _open = open
    with _open(fasta_name) as f:
        fasta_sequences = SeqIO.parse(f, 'fasta')
        for fasta in fasta_sequences:
            if returndescription:
                yield fasta.description, str(fasta.seq)
            elif returnheader:
                yield fasta.id, str(fasta.seq)
            else:
                yield str(fasta.seq)


def load_matrix(path):
    return np.array(pd.read_csv(path, sep=' ', header=None))


def aacounts_str(seq):
    return aacounts_int(map_aatonumber(seq))

@jit(nopython=True)
def aacounts_int_jit(seq):
    counter = np.zeros(len(aminoacids), dtype=np.int64)
    for c in seq:
        counter[c] += 1
    return counter

def to_aacounts(matrix):
    return np.array([list(aacounts_int_jit(seq)) for seq in matrix])


@njit
def frequencies(matrix, num_symbols, pseudocount=0, weights=None):
    """
    Calculate single-site frequencies

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
    num_symbols : int
        Number of different symbols
    weights: np.array
        Vector of length N of relative weights of different sequences

    Returns
    -------
    np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters
    """
    N, L = matrix.shape
    fi = pseudocount/num_symbols * np.ones((L, num_symbols))
    if weights is None:
        for s in range(N):
            for i in range(L):
                fi[i, matrix[s, i]] += 1.0
        return fi / (N+pseudocount)
    else:
        normalized_weights = N*weights/np.sum(weights)
        for s in range(N):
            for i in range(L):
                fi[i, matrix[s, i]] += normalized_weights[s]
    return fi / (N+pseudocount)

@njit
def pair_frequencies(matrix, num_symbols, fi, pseudocount=0, weights=None):
    """
    Calculate pairwise frequencies of symbols.

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
    num_symbols : int
        Number of different symbols
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters.
    weights: np.array
        Vector of length N of relative weights of different sequences

    Returns
    -------
    np.array
        Matrix of size L x L x num_symbols x num_symbols containing
        relative pairwise frequencies of all character combinations
    """
    N, L = matrix.shape
    fij = pseudocount/num_symbols**2 * np.ones((L, L, num_symbols, num_symbols))
    if weights is None:
        for s in range(N):
            for i in range(L):
                for j in range(i + 1, L):
                    fij[i, j, matrix[s, i], matrix[s, j]] += 1
    else:
        normalized_weights = N*weights/np.sum(weights)
        for s in range(N):
            for i in range(L):
                for j in range(i + 1, L):
                    fij[i, j, matrix[s, i], matrix[s, j]] += normalized_weights[s]

    # symmetrize matrix
    for i in range(L):
        for j in range(i + 1, L):
            for alpha in range(num_symbols):
                for beta in range(num_symbols):
                    fij[j, i, beta, alpha] = fij[i, j, alpha, beta]
 
    # normalize frequencies by the number
    # of sequences
    fij /= (N+pseudocount)

    # set the frequency of a pair (alpha, alpha)
    # in position i to the respective single-site
    # frequency of alpha in position i
    for i in range(L):
        for alpha in range(num_symbols):
            fij[i, i, alpha, alpha] = fi[i, alpha]

    return fij

def pair_frequencies_average(fij, copy=False):
    """ Average pair frequencies to make them translation invariant """
    if copy:
        fij = np.copy(fij)
    L = fij.shape[0]
    fij_d = np.array([np.mean(np.array([list(fij[i, i+d]) for i in range(0, L-d)]), axis=0) for d in range(0, L)])
    for i in range(L):
        for j in range(L):
            fij[i, j] = fij_d[np.abs(j-i)]
    return fij

def compute_covariance_matrix(fi, fij):
    cij = fij[:, :, :, :] - fi[:, np.newaxis, :, np.newaxis] * fi[np.newaxis, :, np.newaxis, :]
    return cij


@njit
def compute_flattened_covariance_matrix(fi, fij):
    """
    Compute the covariance matrix in a flat format for mean-field inversion.

    Parameters
    ----------
    fi : np.array
        Matrix of size L x num_symbols
        containing frequencies.
    fij : np.array
        Matrix of size L x L x num_symbols x
        num_symbols containing pair frequencies.

    Returns
    -------
    np.array
        Covariance matrix of size L x (num_symbols-1) x L x (num_symbols-1) 
        
    """
    L, num_symbols = fi.shape
    # The covariance values concerning the last symbol
    # are required to equal zero and are not represented
    # in the covariance matrix (important for taking the
    # inverse) - resulting in a matrix of size
    # (L * (num_symbols-1)) x (L * (num_symbols-1))
    # rather than (L * num_symbols) x (L * num_symbols).
    covariance_matrix = np.zeros((L * (num_symbols - 1),
                                  L * (num_symbols - 1)))
    for i in range(L):
        for j in range(L):
            for alpha in range(num_symbols - 1):
                for beta in range(num_symbols - 1):
                    covariance_matrix[
                        _flatten_index(i, alpha, num_symbols),
                        _flatten_index(j, beta, num_symbols),
                    ] = fij[i, j, alpha, beta] - fi[i, alpha] * fi[j, beta]
    return covariance_matrix

@njit
def _flatten_index(i, alpha, num_symbols):
    """
    Map position and symbol to index in
    the covariance matrix.

    Parameters
    ----------
    i : int, np.array of int
        The alignment column(s).
    alpha : int, np.array of int
        The symbol(s).
    num_symbols : int
        The number of symbols of the
        alphabet used.
    """
    return i * (num_symbols - 1) + alpha