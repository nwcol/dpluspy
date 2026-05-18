"""
For computing ``H_2`` from genotype probability data, stored in GVCF format.
"""

import gzip
import numpy as np

import dpluspy


def compute_sum_H2(rec_map, bins, het_probs, verbose=True):
    """
    Computes the numerator and denominator of the ``H_2`` statistic across an
    array of bins.

    Denominators are computed within this function because coverage is likely to
    vary from sample to sample when data is low-coverage.

    :param rec_map: Array defining the (linear) recombination map coordinates
        of sites in `het_probs`.
    :param bins: Array of recombination-distance bin edges, in the same linear
        map units (Morgans or centiMorgans) as `rec_map`.
    :param het_probs: Array of weights assigned to sites. Weights are the
        probabilities of sampling heterozygous allele copies at each site.
    :param verbose: If True (default), prints updates during computation.

    :returns: Arrays of binned numerators and denominators.
    """
    num_bins = len(bins) - 1
    numer = np.zeros(num_bins, dtype=np.float64)
    denom = np.zeros(num_bins, dtype=np.float64)

    if bins[0] == 0:
        indices_0 = np.arange(1, len(rec_map) + 1)
    else:
        indices_0 = np.searchsorted(rec_map, rec_map + bins[0])

    cum_weights = np.concatenate(([0], np.cumsum(het_probs)))
    cum_sum_0 = cum_weights[indices_0]

    for ii, upper in enumerate(bins[1:]):
        indices_1 = np.searchsorted(rec_map, rec_map + upper)
        denom[ii] = (indices_1 - indices_0).sum()
        cum_sum_1 = cum_weights[indices_1]
        numer[ii] = (het_probs * (cum_sum_1 - cum_sum_0)).sum()
        indices_0 = indices_1
        cum_sum_0 = cum_sum_1
        if verbose:
            print(dpluspy.utils._current_time(),
                  f"locus pairs summed (within) in bin {ii}")
    return numer, denom


def one_sample_H2(rec_map, bins, gp_matrix):
    """

    :param rec_map: Array; not func
    :param bins:
    :param gp_matrix: Array of posterior genotype probabilities 0/0, 0/1, 1/1
    """
    het_probs = gp_matrix[:, 1]
    result = compute_sum_H2(rec_map, bins, het_probs)
    return result


def two_sample_H2(rec_map, bins, gp_matrix_i, gp_matrix_j):
    """
    Docstring for two_sample_stat

    TODO deal with partial non-overlap in coverage between two samples!

    :param rec_map:
    :param bins:
    :param gp_matrix_i: Array of genotype probabilities for sample i
    :param gp_matrix_j: Array of genotype probabilities for sample j
    """
    p_00_i, p_01_i, p_11_i = gp_matrix_i.T
    p_00_j, p_01_j, p_11_j = gp_matrix_j.T
    het_probs = (
        0.5 * p_01_i * p_00_j
        + p_11_i * p_00_j
        + 0.5 * p_00_i * p_01_j
        + 0.5 * p_01_i * p_01_j
        + 0.5 * p_11_i * p_01_j
        + p_00_i * p_11_j
        + 0.5 * p_01_i * p_11_j
    )
    result = compute_sum_H2(rec_map, bins, het_probs)
    return result


def read_vcf(
    vcf_fname,
    bed_fname=None,
    interval=None,
    sample_ids=None,
    apply_filter=False,
    verbose=1e6
):
    """
    TODO write me.

    Note: not for GVCFs, but for typical VCF format files containing all
    covered sites (incl. sites called as monomorphic). Skips multiallelic sites

    :param interval: Upper and lower bound on sits to load. If None (default),
        loads all sites. Intervals are 1-indexed with an inclusive lower and
        exclusive upper bound.
    :param sample_ids: List of samples to load. If None (default), loads all
        samples present.
    :param apply_filter: If True (default False), only load sites with PASS
        in the FILTER column.
    :param verbose:

    :returns:
    """
    counter = 0

    if bed_fname is not None:
        regions, _ = dpluspy.utils._read_bed_file(bed_fname)
        mask = dpluspy.utils._regions_to_mask(regions)
    else:
        mask = None

    if vcf_fname.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    # retrieve indices of desired samples
    with opener(vcf_fname, 'rb') as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    all_ids = line.split()[9:]
                    if sample_ids is None:
                        sample_ids = all_ids
                        sample_idx = list(range(len(all_ids)))
                    else:
                        sample_idx = [all_ids.index(x) for x in sample_ids]
                continue

    # construct data structure
    if interval is not None:
        decrement = interval[0]
    else:
        decrement = 1
    gp_dict = {sample_id: [] for sample_id in sample_ids}
    sites = []
    refs = []
    alts = []

    with opener(vcf_fname, 'rb') as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                continue
            split_line = line.strip().split()
            pos1 = int(split_line[1])
            pos0 = pos1 - decrement

            # check whether the site falls within `interval`
            if interval is not None:
                if pos1 < interval[0]:
                    continue
                if pos1 >= interval[1]:
                    break

            # check whether the site is masked out
            if mask is not None:
                if pos0 >= len(mask):
                    break
                if mask[pos0] == True:
                    continue

            # filter site
            if apply_filter:
                filtr = split_line[6]
                if filtr != "PASS":
                    continue

            ref = split_line[3]
            alt = split_line[4].split(",")
            info = split_line[7]
            fmt = split_line[8].split(":")

            # check to make sure site is a SNP
            alleles = [ref] + alt
            if np.any([len(allele) > 1 for allele in alleles]):
                continue

            # skip multiallelic sites
            if len(alt) > 1:
                continue

            # record ref/alt alleles
            refs.append(ref)
            alts.append(alt[0])

            gp_idx = fmt.index("GP")
            samples = [split_line[9:][idx] for idx in sample_idx]
            split_samples = [sample.split(':') for sample in samples]
            for idx, sample_id in zip(sample_idx, sample_ids):
                probs = split_samples[idx][gp_idx].split(",")
                if len(probs) == 0:
                    raise ValueError("all VCF lines must contain three GP")
                gp_dict[sample_id].append(probs)
            sites.append(pos0)

            if verbose > 1:
                if counter % verbose == 0 and counter > 1:
                    print(dpluspy.utils._current_time(),
                        f'parsed POS {pos1} line {counter}')
            counter += 1

    # turn PL matrices into arrays
    gp_dict = {s: np.array(gp_dict[s], np.float64) for s in gp_dict}
    sites = np.array(sites, np.int64)
    return sites, refs, alts, gp_dict



