"""
Implements several variants of the two-locus coalescent with recombination
"""


import numpy as np
import scipy


_matrix_cache = {}


def equil_cov(rho):
    """
    Compute equilibrium cov(TL, TR) at `rho`
    """
    return (rho + 18) / (rho ** 2 + 13 * rho + 18)


def get_transition_matrix(lam, rr):
    """
    Get the transition matrix in discrete time
    """
    assert lam < 1
    assert rr < 1
    PP = np.array(
        [[0,   2*rr,  0,     0,     0,     0,     0,     0,   lam],
         [lam, 0,     rr,    lam,   lam,   0,     0,     0,   0  ],
         [0,   4*lam, 0,     0,     0,     lam,   lam,   0,   0  ],
         [0,   0,     0,     0,     0,     rr,    0,     0,   lam],
         [0,   0,     0,     0,     0,     0,     rr,    0,   lam],
         [0,   0,     0,     2*lam, 0,     0,     0,     lam, 0  ],
         [0,   0,     0,     0,     2*lam, 0,     0,     lam, 0  ],
         [0,   0,     0,     0,     0,     0,     0,     0,   lam],
         [0,   0,     0,     0,     0,     0,     0,     0,   0, ]],
         np.float64
    )
    PP[np.diag_indices(9)] = 1 - np.sum(PP, axis=1)
    return PP


def discrete_two_locus_coal(lam, rr):
    """
    Run a monte carlo simulation of the coalescent with recombination.
    :param lam: coalescent rate 1/2N
    :param rr: recombination rate (physical)
    :returns: (TL, TR)
    """
    # cache the transition matrix.
    # we make use of the cumulative sum of the matrix on axis 1
    if (lam, rr) in _matrix_cache:
        PP = _matrix_cache[(lam, rr)]
    else:
        _PP = get_transition_matrix(lam, rr)
        PP = np.cumsum(_PP, axis=1)
        _matrix_cache[(lam, rr)] = PP
    state = 0
    TL = None 
    TR = None
    t = 0
    while state not in {7, 8}:
        p = np.random.uniform()
        state = np.searchsorted(PP[state], p)
        if TL is None:
            if state in {4, 6, 7, 8}:
                TL = t
        if TR is None:
            if state in {3, 5, 7, 8}:
                TR = t
        t += 1
    return np.array([TL, TR])


def discrete_noneq_two_locus_coal(lamfunc, rr):
    """
    Uses `lamfunc` to determine the coalescence rate at time t
    """
    state = 0
    TL = None 
    TR = None
    t = 0
    while state not in {7, 8}:
        # instantiate transition matrix or load from cache
        lam = lamfunc(t)
        if (lam, rr) in _matrix_cache:
            PPt = _matrix_cache[(lam, rr)]
        else:
            _PPt = get_transition_matrix(lam, rr)
            PPt = np.cumsum(_PPt, axis=1)
            _matrix_cache[(lam, rr)] = PPt
        p = np.random.uniform()
        state = np.searchsorted(PPt[state], p)
        if TL is None:
            if state in {4, 6, 7, 8}:
                TL = t
        if TR is None:
            if state in {3, 5, 7, 8}:
                TR = t
        t += 1
    return np.array([TL, TR])


def get_intensity_matrix(lam, rho):
    """
    Build an intensitry matrix 
    """
    QQ = np.array(
        [[0,   rho,   0,     0,     0,     0,     0,     0,   lam],
         [lam, 0,     rho/2, lam,   lam,   0,     0,     0,   0  ],
         [0,   4*lam, 0,     0,     0,     lam,   lam,   0,   0  ],
         [0,   0,     0,     0,     0,     rho/2, 0,     0,   lam],
         [0,   0,     0,     0,     0,     0,     rho/2, 0,   lam],
         [0,   0,     0,     2*lam, 0,     0,     0,     lam, 0  ],
         [0,   0,     0,     0,     2*lam, 0,     0,     lam, 0  ],
         [0,   0,     0,     0,     0,     0,     0,     0,   lam],
         [0,   0,     0,     0,     0,     0,     0,     0,   0, ]],
         np.float64
    )
    QQ[np.diag_indices(9)] = -np.sum(QQ, axis=1)
    return QQ


def get_routing_matrix(QQ):
    """
    Compute a routing matrix from some intensity (rate) matrix QQ. 
    :param QQ: Intensity matrix
    """
    diag = -QQ[np.diag_indices(len(QQ))]
    RR = np.zeros(QQ.shape, np.float64)
    RR[:, :] = QQ[:, :]
    np.fill_diagonal(RR, 0)
    RR[diag > 0] /= diag[diag > 0, np.newaxis]
    zeros = np.where(diag == 0)[0]
    RR[zeros, zeros] = 1
    return RR


_matrix_cache = {}


def retrieve_matrices(lam, rho):
    """ 
    Load rate/routing matrices from the cache, or initialize them if they are
    not yet cached.
    """    
    if (lam, rho) in _matrix_cache:
        QQ, RR = _matrix_cache[(lam, rho)]
    else:
        QQ = get_intensity_matrix(lam, rho)
        RR = get_routing_matrix(QQ)
        _matrix_cache[(lam, rho)] = (QQ, RR)
    return QQ, RR


def two_locus_coal(rho, log=False):
    """
    :param rho: 
    """
    # cache the transition matrix.
    # we make use of the cumulative sum of the matrix on axis 1
    lam = 1
    QQ, RR = retrieve_matrices(lam, rho)
    states = np.arange(9)
    state = 0
    TL = None 
    TR = None
    tt = 0
    if log: 
        path = [0]
        times = [0]
    while state != 8:
        tt += np.random.exponential(-1 / QQ[state, state])
        state = np.random.choice(states, p=RR[state])
        if TL is None:
            if state in {4, 6, 7, 8}:
                TL = tt
        if TR is None:
            if state in {3, 5, 7, 8}:
                TR = tt
        if log:
            path.append(state)
            times.append(tt)
    if log:
        record = (np.array(path, np.int64), np.array(times, np.float64))
        ret = (np.array([TL, TR]), record)
    else:
        ret = np.array([TL, TR])
    return ret


def noneq_two_locus_coal(epochs, rho, log=False):
    """
    Requires piecewise-constant population size.
    :epochs: List of tuples (t, rel. size)
    """
    # extract information about the model
    # scaling of the coalescence rate acccounted for elsewhere (change this?)
    assert epochs[-1][1] == 1
    assert epochs[0][0] == 0
    epoch_starts = np.array([t for t, _ in epochs])
    states = np.arange(9)
    state = 0
    TL = None 
    TR = None
    tt = 0
    if log:
        path = [0]
        times = [0]
    while state not in {7, 8}:
        epoch_idx = np.searchsorted(epoch_starts, tt, side="right") - 1

        # assign rate and routing matrices based on epoch
        NN = epochs[epoch_idx][1]
        lam = 1 / NN
        _rho = rho * NN
        QQt, RRt = retrieve_matrices(lam, _rho)
        exp_scale = -1 / QQt[state, state]

        # check whether we are in the earliest epoch
        # yes;
        if epoch_idx == len(epochs) - 1:
            tt += np.random.exponential(exp_scale)
            state = np.random.choice(states, p=RRt[state])

        # no;
        else:
            next_epoch_start = epochs[epoch_idx + 1][0]
            time_left = next_epoch_start - tt
            draw = np.random.exponential(exp_scale)
            if draw < time_left:
                tt += draw
                state = np.random.choice(states, p=RRt[state])
            else:
                # tt += time_left
                tt = next_epoch_start
                continue
            
        if TL is None:
            if state in {4, 6, 7, 8}:
                TL = tt
        if TR is None:
            if state in {3, 5, 7, 8}:
                TR = tt

        if log:
            path.append(state)
            times.append(tt)
    if log:
        ret = (np.array([TL, TR]),
               (np.array(path, np.float64), np.array(times, np.float64)))
    else:
        ret = np.array([TL, TR])
    return ret


def compute_state_times(log):
    """
    compute time spent in each state
    """
    states, times = log
    _states = states[:-1]
    holding_times = np.diff(times)
    state_times = np.zeros(8, np.float64)
    for ii in range(8):
        state_times[ii] = holding_times[_states == ii].sum()
    return state_times




class Gamete(tuple):

    def __init__(self, ):

        pass

    @staticmethod
    def coalesce(gamete, gamete_):

        pass

    def recombine(self):

        pass


class Process():

    def __init__(self, ):

        pass
    
    @classmethod
    def initialize(cls, ):
        pass

    def simulate(self):

        pass


class Epoch():

    def __init__(self):
        pass


