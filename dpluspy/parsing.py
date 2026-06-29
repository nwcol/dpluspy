"""


"""

import numpy as np

from .matrices import HaplotypeMatrix, GenotypeMatrix, GPMatrix
from . import util


# =============================================================================
# The principal function for computing H2
# =============================================================================


def compute_h2_stats(
    vcf_file=None,
    haplotype_matrix=None,
    genotype_matrix=None,
    gp_matrix=None,
    pop_file=None,
    pops=None,
    rec_map_file=None,
    r_bins=None,
    bp_bins=None,
    min_bp=None,
    mut_file=None,
    u_bar=None,
    use_genotypes=True,
    use_gps=False,
    report=True,
    bed_file=None,
    interval=None,
    parse_denominator=True,
    pairwise=True
    ):
    """
    Compute H2 statistics on a chromosome or arbitrary chromosome interval and
    return them in a dict.

    Parameters
    ----------
    vcf_file : str, optional
        Path to VCF file. Not needed if haplotype/genotype/genotype prob.
        matrix is given,
    use_gps : bool, optional
        If True (default False), use genotype probabilities to compute H2.
        Raise an error if no VCF file is given.

    Returns
    -------
    dict with 'bins', 'sums', 'stats', 'pops', and optionally 'denoms'.
    """


    # Print information ....

    return


def _compute_h2_sums():
    """
    """
    
    if pairwise is True:
        

    else:
        raise ValueError("not implemented")
    return sums


def compute_h2_denoms(

    ):
    return


def _compute_binned_denoms(coords, bins):
    """Compute binned denominator for two-locus statistics"""
    binned_denoms = np.zeros(len(bins) - 1, dtype=np.float64)
    # Indices (inclusive) of lowest-indexed sites in the zeroth bin
    lower_sites = np.maximum(np.searchsorted(coords, coords + bins[0]),
                             np.arange(1, len(coords) + 1))
    for ii, upper_edge in enumerate(bins[1:]):
        # Indices (exclusive) of highest-indexed sites in bin `ii`
        upper_sites = np.searchsorted(coords, coords + upper_edge)
        binned_denoms[ii] = np.sum(upper_sites - lower_sites)
        lower_sites = upper_sites
    return binned_denoms


# -----------------------------------------------------------------------------
# Functions for averaging/bootstrapping across genomic intervals
# -----------------------------------------------------------------------------


def mean_across_data():

    return


def draw_boostrap_sets():

    return


def bootstrap_data():

    return


def subset_data():

    return


# -----------------------------------------------------------------------------
# Pairwise estimators- each operates on a pair of diploids
# -----------------------------------------------------------------------------


def _call_pairwise_estimator(matrix, samples, coords, bins, weights=None):
    """
    Compute H2 for a diploid or pair of diploids.

    Parameters
    ----------
    matrix : HaplotypeMatrix, GenotypeMatrix, or GPMatrix
    samples : 2-tuple/list or int
    coords : np.ndarray
        Shape (n_sites).
    bins : np.ndarray
        Shape (n_bins + 1). Defines bin edges, in the same unit as `coords`.
    weights : np.ndarray, optional
        Shape (n_sites). Specifies relative weights for each site.

    Returns
    -------
    np.ndarray of binned H2 sums.
    """
    if isinstance(samples, int):
        samples = (samples, samples)

    sample1, sample2 = samples

    # Single-diploid
    if sample1 == sample2:
        if isinstance(matrix, HaplotypeMatrix):
            start = 2 * sample1
            end = 2 * sample1 + 2
            haplotypes = matrix.haplotypes[:, start:end]
            sums = _h2_pw_haplotype_within(
                haplotypes, coords, bins, weights=weights)

        elif isinstance(matrix, GenotypeMatrix):
            genotypes = matrix.genotypes[:, sample1]
            sums = _h2_pw_genotype_within(
                genotypes, coords, bins, weights=weights)

        elif isinstance(matrix, GenoProbMatrix):
            start = 3 * sample1
            end = 3 * sample1 + 3
            probs = matrix.probs[:, start:end]
            sums = _h2_pw_genoprob_within(probs, coords, bins, weights=weights)

        else:
            raise ValueError("given matrix class is unsupported")

    # Between-diploid
    else:
        if isinstance(matrix, HaplotypeMatrix):
            start1, start2 = 2 * sample1, 2 * sample2
            end1, end2 = 2 * sample1 + 2, 2 * sample2 + 2
            haplotypes1 = matrix.haplotypes[:, start1:end1]
            haplotypes2 = matrix.haplotypes[:, start2:end2]
            sums = _h2_pw_haplotype_between(
                haplotypes1, haplotypes2, coords, bins, weights=weights)

        elif isinstance(matrix, GenotypeMatrix):
            genotypes1 = matrix.genotypes[:, sample1]
            genotypes2 = matrix.genotypes[:, sample2]
            sums = _h2_pw_genotype_between(
                genotypes1, genotypes2, coords, bins, weights=weights)

        elif isinstance(matrix, GenoProbMatrix):
            start1, start2 = 3 * sample1, 3 * sample2
            end1, end2 = 3 * sample1 + 3, 3 * sample2 + 3
            genoprobs1 = matrix.probs[:, start1:end1]
            genoprobs2 = matrix.probs[:, start2:end2]
            sums = _h2_pw_genoprob_between(
                genoprobs1, genoprobs1, coords, bins, weights=weights)

        else:
            raise ValueError("given matrix class is unsupported")
    return sums


# Pairwise estimators operate on bare numpy arrays.


def _h2_pw_haplotype_within(haplotypes, coords, bins, weights=None):
    """Compute within-diploid H2 from haplotype (phased) data"""
    is_het = 1.0 * (haplotypes[:, 0] != haplotypes[:, 1])
    if weights is not None:
        is_het = is_het * weights
    return = _compute_binned_sums(is_het, coords, bins)


def _h2_pw_haplotype_between(
    haplotypes1,
    haplotypes2,
    coords,
    bins,
    weights=None
    ):
    """Compute between-diploid H2 from haplotype data"""
    sums = 0.0
    # Average across haplotype-by-haplotype pairs
    for hap1 in haplotypes1.T:
        for hap2 in haplotypes2.T:
            is_het = 1.0 * (hap1 != hap2)
            if weights is not None:
                is_het = weights * is_het
                sums += _compute_binned_sums(is_het, coords, bins)
    return sums / 4


def _h2_pw_genotype_within(genotypes, coords, bins, weights=None):
    """Compute within-diploid H2 from genotype (unphased) data."""
    is_het = 1.0 * (genotypes == 1)
    if weights is not None:
        is_het = weights * is_het
    return _compute_binned_sums(is_het, coords, bins)


def _h2_pw_genotype_between(
    genotypes1,
    genotypes2,
    coords,
    bins,
    weights=None
    ):
    """Compute between-diploid H2 from genotype data"""
    pi_12 = np.abs(genotypes1 - genotypes2) / 2
    if weights is not None:
        pi_12 = weights * pi_12
    return _compute_binned_sums(pi_12, coords, bins)


def _h2_pw_genoprob_within(probs, coords, bins, weights=None):
    """Compute within or single-diploid H2 from genotype probabilities."""
    p_het = probs[:, 1]
    if weights is not None:
        p_het = weights * p_het
    return _compute_binned_sums(p_het, coords, bins)


def _h2_pw_genoprob_between(
    probs1,
    probs2,
    coords,
    bins,
    weights=None
    ):
    """Compute between-diploid H2 from genotype probabilities."""
    p_aa_1, p_aA_1, p_AA_1 = probs1.T
    p_aa_2, p_aA_2, p_AA_2 = probs2.T
    pi_12 = (
        0.5 * p_aa_1 * p_aA_2
        + p_aa_1 * p_AA_2
        + 0.5 * p_aA_1 * p_aa_2
        + 0.5 * p_aA_1 * p_aA_2
        + 0.5 * p_aA_1 * p_AA_2
        + p_AA_1 * p_aa_2
        + 0.5 * p_AA_1 * p_aA_2
        )
    if weights is not None:
        pi_12 = weights * pi_12
    return _compute_binned_sums(pi_12, coords, bins)


def _block_pairs():

    return


def _compute_binned_sums(site_vals, coords, bins):
    """
    Engine for calculating binned sums of one/two-diploid H2 estimators across
    locus pairs.

    Specifically, for each of the (n_sites choose 2) pairs of sites i, j that
    can be drawn from `site_vals`, increment `site_vals[i] * site_vals[j]` to
    the bin corresponding to the between-site distance `coords[j] - coords[i]`.

    Within and between-diploid estimators for H2 present a special case, where
    we can usually calculate H2 by taking products of some site value. For a
    single diploid, this value is a boolean indicator for heterozygosity.
    """
    binned_sums = np.zeros(len(bins) - 1, dtype=np.float64)
    lower_sites = np.maximum(np.searchsorted(coords, coords + bins[0]),
                             np.arange(1, len(coords) + 1))
    cumulative = np.concatenate(([0], np.cumsum(site_vals)))
    lower_cum_vals = cumulative[lower_sites]
    for ii, upper_edge in enumerate(bins[1:]):
        upper_sites = np.searchsorted(coords, coords + upper)
        upper_cum_vals = cumulative[upper_sites]
        bin_cum_vals = upper_cum_vals - lower_cum_vals
        # Take the product of left site values with cumulative right site
        # values, then sum across left sites to obtain the bin-wide sum
        binned_sums[ii] = np.sum(site_vals * bin_cum_vals)
        lower_sites = upper_sites
        lower_cum_vals = upper_cum_vals
    return binned_sums


# -----------------------------------------------------------------------------
# Slow counting functions for multi-sample estimation
# -----------------------------------------------------------------------------


def tally_haplotype_pairs(
    haplotypes,
    idx_i=None,
    idx_j=None,
    sample_indices=None
    ):
    """
    Compute two-locus haplotype counts for given locus pairs.

    Parameters
    ----------
    haplotypes : np.ndarray
        Shape (n_loci, n_haplotypes). Values 0, 1.

    idx_i, idx_j : list
        Length n_pairs. Lists specifying locus pairs. When not given,
        all locus pairs are included.

    sample_indices : list
        Indices of haplotypes to include.
    """
    if sample_indices is None:
        sample_indices = list(range(haplotypes.shape[1]))
    haplotypes = haplotypes[:, sample_indices]

    if idx_i is None and idx_j is None:
        n_loci = genotypes.shape[0]
        idx_i = [i for i in range(n_loci) for j in range(i + 1, n_loci)]
        idx_j = [j for i in range(n_loci) for j in range(i + 1, n_loci)]
    else:
        assert idx_i is not None and idx_j is not None

    locus_i = haplotypes[idx_i]
    locus_j = haplotypes[idx_j]

    n11 = np.sum((locus_i == 1) & (locus_j == 1), axis=1)
    n10 = np.sum((locus_i == 1) & (locus_j == 0), axis=1)
    n01 = np.sum((locus_i == 0) & (locus_j == 1), axis=1)
    n00 = np.sum((locus_i == 0) & (locus_j == 0), axis=1)

    counts = np.stack([n11, n10, n01, n00], axis=1)
    return counts


def tally_genotype_pairs(
    genotypes,
    idx_i=None,
    idx_j=None,
    sample_indices=None
    ):
    """
    Compute two-locus genotype counts for given locus pairs.

    Not very efficient.

    Parameters
    ----------
    genotypes : np.ndarray
        Shape (n_loci, n_samples). Values 0, 1, 2.

    idx_i, idx_j : list
        Length n_pairs. Lists specifying locus pairs. When not given,
        all locus pairs are included.

    sample_indices : list
        Indices of genotypes to include.
    """
    if sample_indices is None:
        sample_indices = list(range(genotypes.shape[1]))
    genotypes = genotypes[:, sample_indices]

    if idx_i is None and idx_j is None:
        n_loci = genotypes.shape[0]
        idx_i = [i for i in range(n_loci) for j in range(i + 1, n_loci)]
        idx_j = [j for i in range(n_loci) for j in range(i + 1, n_loci)]
    else:
        assert idx_i is not None and idx_j is not None

    locus_i = genotypes[idx_i]
    locus_j = genotypes[idx_j]

    n22 = np.sum((locus_i == 2) & (locus_j == 2), axis=1)
    n21 = np.sum((locus_i == 2) & (locus_j == 1), axis=1)
    n20 = np.sum((locus_i == 2) & (locus_j == 0), axis=1)
    n12 = np.sum((locus_i == 1) & (locus_j == 2), axis=1)
    n11 = np.sum((locus_i == 1) & (locus_j == 1), axis=1)
    n10 = np.sum((locus_i == 1) & (locus_j == 0), axis=1)
    n02 = np.sum((locus_i == 0) & (locus_j == 2), axis=1)
    n01 = np.sum((locus_i == 0) & (locus_j == 1), axis=1)
    n00 = np.sum((locus_i == 0) & (locus_j == 0), axis=1)

    counts = np.stack([n22, n21, n20, n12, n11, n10, n02, n01, n00], axis=1)
    return counts


def compute_expected_two_locus_genotypes(
    gprobs,
    idx_i=None,
    idx_j=None,
    sample_indices=None
    ):
    """
    Compute expected tallies of two-locus genotypes from genotype probabilities.

    Parameters
    ----------
    gprobs : np.ndarray
        Shape (n_loci, 3 * n_samples). For sample i, columns i, i + 1, i + 2 hold
        the posterior probabilities assigned to genotypes 0/0, 0/1, 1/1.
    """
    if sample_indices is None:
        sample_indices = list(range(gprobs.shape[1]))
    gprobs = gprobs[:, sample_indices]

    if idx_i is None and idx_j is None:
        n_loci = gprobs.shape[0]
        idx_i = [i for i in range(n_loci) for j in range(i + 1, n_loci)]
        idx_j = [j for i in range(n_loci) for j in range(i + 1, n_loci)]
    else:
        assert idx_i is not None and idx_j is not None

    locus_i = gprobs[idx_i]
    locus_j = gprobs[idx_j]

    n22 = np.sum(locus_i[:, 2::3] * locus_j[:, 2::3], axis=1)
    n21 = np.sum(locus_i[:, 2::3] * locus_j[:, 1::3], axis=1)
    n20 = np.sum(locus_i[:, 2::3] * locus_j[:, ::3], axis=1)
    n12 = np.sum(locus_i[:, 1::3] * locus_j[:, 2::3], axis=1)
    n11 = np.sum(locus_i[:, 1::3] * locus_j[:, 1::3], axis=1)
    n10 = np.sum(locus_i[:, 1::3] * locus_j[:, ::3], axis=1)
    n02 = np.sum(locus_i[:, ::3] * locus_j[:, 2::3], axis=1)
    n01 = np.sum(locus_i[:, ::3] * locus_j[:, 1::3], axis=1)
    n00 = np.sum(locus_i[:, ::3] * locus_j[:, ::3], axis=1)

    exp_counts = np.stack([n22, n21, n20, n12, n11, n10, n02, n01, n00], axis=1)
    return exp_counts


# -----------------------------------------------------------------------------
# Multi-sample estimators- these take precomputed genotype/haplotype counts
# -----------------------------------------------------------------------------


def _xxxx():
    """
    Compute H2 sums
    """

    return


def _call_multi_sample_estimator():

    return


def h2_haplotype_within(counts, pop_idx):
    """Calculate within-population H2 from two-locus haplotype counts"""
    start = 4 * pop_idx
    c1, c2, c3, c4 = counts[:, start:start + 4].T
    n = np.sum(counts[start:start + 4], axis=1)
    numer = c1 * c4 + c2 * c3
    denom = n * (n - 1) / 2
    stat = numer / denom
    return stat


def h2_haplotype_between(counts, pop1_idx, pop2_idx):
    """Calculate between-population H2 from two-locus haplotype counts"""
    start1 = pop1_idx * 4
    start2 = pop2_idx * 4
    c11, c12, c13, c14 = counts[:, start1:start1 + 4].T
    c21, c22, c23, c24 = counts[:, start2:start2 + 4].T
    n1 = np.sum(counts[:, start1:start1 + 4])
    n2 = np.sum(counts[:, start2:start2 + 4])
    numer = c11 * c24 + c21 * c14 + c12 * c23 + c22 * c13
    denom = n1 * n2
    stat = numer / denom
    return stat


def h2_genotype_within(counts, pop_idx):
    """Compute within-population H2 from genotype counts"""
    start = 9 * pop_idx
    g1, g2, g3, g4, g5, g6, g7, g8, g9 = counts[:, start:start + 9].T
    n = np.sum(counts[:, start:start + 9], axis=1)
    numer = (
        g1 * g5
        + 2 * g1 * g6
        + 2 * g1 * g8
        + 4 * g1 * g9
        + g2 * g4
        + g2 * g5
        + g2 * g6
        + 2 * g2 * g7
        + 2 * g2 * g8
        + 2 * g2 * g9
        + 2 * g3 * g4
        + g3 * g5
        + 4 * g3 * g7
        + 2 * g3 * g8
        + g4 * g5
        + 2 * g4 * g6
        + g4 * g8
        + 2 * g4 * g9
        + g5 * (g5 + 1) / 2
        + g5 * g6
        + g5 * g7
        + g5 * g8
        + g5 * g9
        + 2 * g6 * g7
        + g6 * g8
        )
    denom = n * (2 * n - 1)
    stat = numer / denom
    return stat


def h2_genotype_between(counts, pop1_idx, pop2_idx):
    """Compute between-sample H2 from genotype counts"""
    start1 = 9 * pop1_idx
    start2 = 9 * pop2_idx
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = counts[:, start1:start1 + 9].T
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = counts[:, start2:start2 + 9].T
    n1 = np.sum(counts[:, start1:start1 + 9], axis=1)
    n2 = np.sum(counts[:, start2:start2 + 9], axis=1)
    numer = (
        (g11 + g12 / 2 + g14 / 2 + g15 / 4)
        * (g25 / 4 + g26 / 2 + g28 / 2 + g29)
        + (g15 / 4 + g16 / 2 + g18 / 2 + g19)
        * (g21 + g22 / 2 + g24 / 2 + g25 / 4)
        + (g12 / 2 + g13 + g15 / 4 + g16 / 2)
        * (g24 / 2 + g25 / 4 + g27 + g28 / 2)
        + (g14 / 2 + g15 / 4 + g17 + g18 / 2)
        * (g22 / 2 + g23 + g25 / 4 + g26 / 2)
        )
    denom = n1 * n2
    stat = numer / denom
    return stat


# -----------------------------------------------------------------------------
# Heterozygosity statistics
# -----------------------------------------------------------------------------


def _compute_heterozygosity():


    return


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def h_names(n_pops):
    names = []
    for ii in range(n_pops):
        for jj in range(ii, n_pops):
            names.append(f"H_{ii}_{jj}")
    return names


def h2_names(n_pops):
    names = []
    for ii in range(n_pops):
        for jj in range(ii, n_pops):
            names.append(f"H2_{ii}_{jj}")
    return names

