"""
Simple recursions of the D+ statistic
"""


import numpy as np 


def HR_transition_matrix(u, r, N):
    """
    This matrix provides rates for the basis [1, Hl, Hr, D^2, Dz, pi2, D+].

    :param N: Diploid population size.

    Time is measured in generations.
    """
    # Indices
    I = 0
    Hl = 1
    Hr = 2
    DD = 3
    Dz = 4
    pi2 = 5
    Dp = 6
    lam = 1 / (2 * N)
    Mut = np.zeros((7, 7))
    Mut[Hl, I] = u
    Mut[Hr, I] = u
    Mut[pi2, Hl] = u
    Mut[pi2, Hr] = u
    Mut[Dp, Hl] = 2 * u  # D+ created through u
    Mut[Dp, Hr] = 2 * u

    Rec = np.zeros((7, 7))
    Rec[DD, DD] = -2 * r  # D^2 decay
    Rec[Dz, Dz] = -1 * r
    Rec[Dp, DD] = -8 * r
    Rec[Dp, Dz] = -2 * r

    Drift = np.zeros((7, 7))
    Drift[Hl, Hl] = -lam
    Drift[Hr, Hr] = -lam
    Drift[DD, DD] = -3 * lam
    Drift[DD, Dz] = 1 * lam 
    Drift[DD, pi2] = 1 * lam 
    Drift[Dz, DD] = 4 * lam
    Drift[Dz, Dz] = -5 * lam 
    Drift[pi2, Dz] = 1 * lam
    Drift[pi2, pi2] = -2 * lam
    Drift[Dp, Dp] = -lam # Decay of D+ through drift

    M = Mut + Rec + Drift
    v = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    return v, M


