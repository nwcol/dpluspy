"""
Odds and ends for comprehending some theoretical aspects of D+ and its 
evolution (nothing systematic). 
"""

import numpy as np
from sympy import *


## Bases


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



## Sympy representations 


def Hill_Robertson():   

    c1, c2, c3, c4, p, q = symbols("c1 c2 c3 c4 p q")
    D, DD, pi2, z, Dz = symbols("D DD pi2 z Dz")
    p = c1 + c2
    q = c1 + c3
    D = c1 * c4 - c2 * c3
    DD = c1 ** 2 * c4 ** 2 - 2 * c1 * c2 * c3 * c4 + c2 ** 2 * c3 ** 2
    z = (c3 + c4 - c1 - c2) * (c2 + c4 - c1 - c3)
    Dz = D * z
    pi2 = (c1 + c2) * (c3 + c4) * (c1 + c3) * (c2 + c4)
    4 * DD + 2 * Dz + 4 * pi2
    return


def genotype_estimator_one_pop():
    """

    Haplotype defs. estimators labelled by x
        B   b
    A   1   2
    a   3   4

    Genotype defs. counts labelled by n
        BB  Bb  bb
    AA  1   2   3
    Aa  4   5   6
    aa  7   8   9
    
    """
    # let `n1` count nenotypes
    # initialize symbols
    n1, n2, n3, n4, n5, n6, n7, n8, n9 = symbols("n1 n2 n3 n4 n5 n6 n7 n8 n9")
    hat_dp, n, x1, x2, x3, x4 = symbols("D+ n x1 x2 x3 x4")
    # define haplotype estimators
    x1 = n1 + n2/2 + n4/2 + n5/4
    x2 = n3 + n2/2 + n6/2 + n5/4
    x3 = n7 + n4/2 + n8/2 + n5/4
    x4 = n9 + n6/2 + n8/2 + n5/4
    assert x1 + x2 + x3 + x4 == n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9
    hat_dp = 2 * x1 * x4 + 2 * x2 * x3
    expanded = expand(hat_dp)
    monomes = str(expanded).split(" + ") 
    for i, monome in enumerate(monomes):
        if i == 0:
            print("D+ = " + monome)
        else: 
            print("    + " + monome)

    return


def genotype_estimator_two_pop():

    ni1, ni2, ni3, ni4, ni5, ni6, ni7, ni8, ni9 = symbols(
        "n_{i1} n_{i2} n_{i3} n_{i4} n_{i5} n_{i6} n_{i7} n_{i8} n_{i9}")
    nj1, nj2, nj3, nj4, nj5, nj6, nj7, nj8, nj9 = symbols(
        "n_{j1} n_{j2} n_{j3} n_{j4} n_{j5} n_{j6} n_{j7} n_{j8} n_{j9}")
    xi1, xi2, xi3, xi4, xj1, xj2, xj3, xj4 = symbols(
        "x_{i1} x_{i2} x_{i3} x_{i4} x_{j1} x_{j2} x_{j3} x_{j4}")
    hat_dp12, n1, n2 = symbols("D+12 n1 n2")
    # define haplotype estimators
    xi1 = ni1 + ni2/2 + ni4/2 + ni5/4
    xi2 = ni3 + ni2/2 + ni6/2 + ni5/4
    xi3 = ni7 + ni4/2 + ni8/2 + ni5/4
    xi4 = ni9 + ni6/2 + ni8/2 + ni5/4
    xj1 = nj1 + nj2/2 + nj4/2 + nj5/4
    xj2 = nj3 + nj2/2 + nj6/2 + nj5/4
    xj3 = nj7 + nj4/2 + nj8/2 + nj5/4
    xj4 = nj9 + nj6/2 + nj8/2 + nj5/4
    assert xi1 + xi2 + xi3 + xi4 == ni1 + ni2 + ni3 + ni4 + ni5 + ni6 + ni7 + ni8 + ni9
    assert xj1 + xj2 + xj3 + xj4 == nj1 + nj2 + nj3 + nj4 + nj5 + nj6 + nj7 + nj8 + nj9
    hat_dp_ij = xi1 * xj4 + xj1 * xi4 + xi2 * xj3 + xj2 * xi3 
    expanded = expand(hat_dp12)
    monomes = str(expanded).split(" + ")
    for i, monome in enumerate(monomes):
        if i == 0:
            print("D+12 = " + monome)
        else: 
            print("    + " + monome)

    return


## Parsing functions; we don't use these because we do not actually count
## haplotypes when parsing.


def tally_haplotypes(h1, h2):
    #
    n = len(h1)
    n11 = (h1 & h2).sum()
    n10 = h1.sum() - n11
    n01 = h2.sum() - n11
    n00 = n - n11 - n10 - n01
    return (n11, n10, n01, n00)


def tally_genotypes(g1, g2):
    # tallies up two-locus genotypes
    # note that & is a symbol for np.logical_and()
    n = len(g1)
    n22 = ((g1 == 2) & (g2 == 2)).sum()
    n21 = ((g1 == 2) & (g2 == 1)).sum() 
    n20 = (g1 == 2).sum() - n22 - n21
    n12 = ((g1 == 1) & (g2 == 2)).sum()
    n11 = ((g1 == 1) & (g2 == 1)).sum()
    n10 = (g1 == 1).sum() - n12 - n11
    n02 = ((g1 == 0) & (g2 == 2)).sum()
    n01 = ((g1 == 0) & (g2 == 1)).sum()
    n00 = n - n22 - n21 - n20 - n12 - n11 - n10 - n02 - n01
    return (n22, n21, n20, n12, n11, n10, n02, n01, n00)


def _haplotype_h2_from_counts(counts):
    #
    c1, c2, c3, c4 = counts
    numer = c1 * c4 + c2 * c3
    num = c1 + c2 + c3 + c4
    h2 = numer / (num * (num - 1) / 2)    
    return h2


def _two_pop_haplotype_h2_from_counts(counts1, counts2):
    #
    c11, c12, c13, c14 = counts1 
    c21, c22, c23, c24 = counts2 
    numer = c11 * c24 + c14 * c21 + c12 * c23 + c13 * c22
    num1 = c11 + c12 + c13 + c14
    num2 = c21 + c22 + c23 + c24
    h2 = numer / (num1 * num2)
    return h2


def _genotype_h2_from_counts(counts):
    # shape (9, b). counts of two-locus genotypes 1/1-1/1, ... 0/0-0/0
    n1, n2, n3, n4, n5, n6, n7, n8, n9 = counts
    numer = (
        n1 * n5 / 4
        + n1 * n6 / 2
        + n1 * n8 / 2
        + n1 * n9
        + n2 * n4 / 4
        + n2 * n5 / 4
        + n2 * n6 / 4
        + n2 * n7 / 2
        + n2 * n8 / 4
        + n2 * n9 / 2
        + n3 * n4 / 2
        + n3 * n5 / 4
        + n3 * n7
        + n3 * n8 / 2
        + n4 * n5 / 4
        + n4 * n6 / 2
        + n4 * n8 / 4
        + n4 * n9 / 2
        + n5  ######
        + n5 * n6 / 4 
        + n5 * n7 / 4
        + n5 * n8 / 4
        + n5 * n9 / 4
        + n6 * n7 / 2
        + n6 * n8 / 4
    )
    num = counts.sum(0) 
    h2 = numer / (num * (num - 1) / 2)  
    return h2


def _two_pop_genotype_h2_from_counts(counts1, counts2):
    # shapes (9, b)
    n11, n12, n13, n14, n15, n16, n17, n18, n19 = counts1
    n21, n22, n23, n24, n25, n26, n27, n28, n29 = counts2
    numer = (
        n11 * n29 + n19 * n21
        + (n11 * n26 + n16 * n21) / 2
        + (n11 * n28 + n18 * n21) / 2
        + (n11 * n29 + n19 * n21) / 2
        + (n12 * n24 + n14 * n22) / 4
        + (n12 * n25 + n15 * n22) / 4
        + (n12 * n26 + n16 * n22) / 4
        + (n12 * n27 + n17 * n22) / 2
        + (n12 * n28 + n18 * n22) / 4
        + (n12 * n29 + n19 * n22) / 2
        + (n13 * n24 + n14 * n23) / 2
        + (n13 * n25 + n15 * n23) / 4
        + n13 * n27 + n17 * n23
        + (n13 * n28 + n18 * n23) / 2
        + (n14 * n25 + n15 * n24) / 4
        + (n14 * n26 + n16 * n24) / 2
        + (n14 * n28 + n18 * n24) / 4
        + (n14 * n29 + n19 * n24) / 2
        + (n15 * n25) / 4 
        + (n15 * n26 + n16 * n25) / 4 
        + (n15 * n27 + n17 * n25) / 4
        + (n15 * n28 + n19 * n25) / 4
        + (n15 * n29 + n19 * n25) / 4
        + (n16 * n27 + n17 * n26) / 2
        + (n16 * n28 + n18 * n26) / 4
    )
    num1 = counts1.sum(0)
    num2 = counts1.sum(0)
    h2 = numer / (num1 * num2)
    return h2
