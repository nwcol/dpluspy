"""
Functions for computing D+ (and H) from sequence data
"""

from datetime import datetime
import gzip
import io
import numpy as np
import pandas
import re
import scipy
import warnings

import dpluspy


def parse_stats(
    vcf_file,
    u_bar=1e-8,
    ts_sample_ids=None,
    bed_file=None,
    label_file=None,
    labels=None,
    rec_map_file=None,
    pos_col="Position(bp)",
    map_col="Map(cM)",
    map_sep=None,
    interp_method="linear",
    r=None,
    r_bins=None,
    bp_bins=None,
    mut_map_file=None,
    mut_col=None,
    intervals=None,
    interval_file=None,
    chrom="None",
    phased=False,
    get_cross_pop=True,
    get_denoms=True,
    allow_multi=True,
    missing_to_ref=False,
    apply_filter=False,
    verbose=True
):
    """

    """
    # Load interval file
    if interval_file is not None:
        intervals = np.loadtxt(interval_file)
    
    if intervals is None:
        raise ValueError("You must provide `intervals` or `interval_file`")

    if np.array(intervals).ndim == 1:
        intervals = np.array(intervals)[None, :]

    # Convert intervals into a list of 1d arrays
    intervals = [np.asarray(x).flatten().astype(np.int64) for x in intervals]

    if get_denoms:
        if bed_file is not None: 
            positions = dpluspy.utils._read_bed_file_positions(bed_file) + 1
            seq_length = positions[-1]
        else:
            seq_length = intervals[-1][-1]
            positions = np.arange(1, seq_length)
    else:
        positions = None
        seq_length = intervals[-1][-1] + 1

    if r_bins is not None:
        if isinstance(r_bins, str):
            r_bins = np.loadtxt(r_bins)
        # Convert bins in r to Morgans 
        bins = dpluspy.utils._map_function(r_bins)
        if rec_map_file is not None:
            map_fxn = _load_recombination_map(
                rec_map_file, 
                pos_col=pos_col,
                map_col=map_col,
                interp_method=interp_method,
                map_sep=map_sep
            )
        elif r is not None:
            map_fxn = _get_uniform_recombination_map(r, seq_length)
        else:
            raise ValueError("You must provide recombination map information")
        
        # Save r bins for output
        ret_bins = r_bins
    elif bp_bins is not None:
        map_fxn = lambda x: x
        ret_bins = None
    else:
        raise ValueError("You must provide bins")

    if label_file is not None:
        labels = load_label_file(label_file)

    if labels is not None:
        vcf_samples = [labels[label] for label in labels]
    else:
        vcf_samples = None

    # Read genotypes from a VCF file or extract them from a tree sequence
    if isinstance(vcf_file, str):
        sites, genotypes, sample_ids = get_vcf_genotypes(
            vcf_file, 
            sample_ids=vcf_samples,
            bed_file=bed_file, 
            allow_multi=allow_multi,
            missing_to_ref=missing_to_ref,
            apply_filter=apply_filter
        )
    else:
        sites, genotypes, sample_ids = get_ts_genotypes(
            vcf_file, 
            ts_sample_ids=ts_sample_ids,
            sample_ids=vcf_samples,
            bed_file=bed_file, 
            allow_multi=allow_multi,
            missing_to_ref=missing_to_ref,
            apply_filter=apply_filter
        )

    # Construct a dict mapping population IDs to population genotype arrays
    genotype_dict = get_genotype_dict(
        genotypes, sample_ids, sample_labels=labels)

    # Load the mutation map, if one was provided
    if mut_map_file is not None:
        mut_map = _load_mutation_map(mut_map_file, sites, map_col=mut_col)
    else:
        mut_map = None
    
    # Compute statistics!
    stats = compute_stats(    
        sites,
        genotype_dict,
        map_fxn,
        bins,
        intervals,
        positions=positions,
        mut_map=mut_map,
        u_bar=u_bar,
        chrom=chrom,
        get_cross_pop=get_cross_pop,
        phased=phased,
        ret_bins=ret_bins,
        verbose=verbose
    )
    return stats


def compute_stats(
    sites,
    genotype_dict,
    map_func,
    bins,
    intervals,
    mut_map=None,
    u_bar=None,
    positions=None,
    chrom="None",
    get_cross_pop=True,
    phased=False,
    verbose=True,
    ret_bins=None
):
    """
    
    """
    samples = list(genotype_dict.keys())
    ret = dict()

    for ii, interval in enumerate(intervals):
        assert len(interval) == 3
        left_interval = interval[:2]
        right_interval = interval[1:]
        stats = dict()
        stats["bins"] = ret_bins
        stats["pop_ids"] = samples

        if positions is not None:
            stats["denoms"] = denoms_within(
                positions, map_func, bins, left_interval)
        stats["sums"] = get_stats_within(
            sites, 
            left_interval, 
            genotype_dict, 
            map_func, 
            bins, 
            mut_map=mut_map,
            u_bar=u_bar,
            get_cross_pop=get_cross_pop,
            phased=phased
        )
        if verbose:
            print(_current_time(), 
                f"Computed stats within chrom {chrom} interval {ii} "
                f"{interval[0]}-{interval[1]}")
        
        if right_interval[1] > right_interval[0]:
            if positions is not None:
                stats["denoms"] += denoms_between(
                    positions, map_func, bins, (left_interval, right_interval))
            stats["sums"] += get_stats_between(
                sites, 
                (left_interval, right_interval), 
                genotype_dict, 
                map_func, 
                bins,
                mut_map=mut_map,
                u_bar=u_bar,
                get_cross_pop=get_cross_pop,
                phased=phased
            )
            if verbose:
                print(_current_time(), 
                    f"Computed stats between chrom {chrom} intervals {ii} "
                    f"{left_interval[0]}-{right_interval[1]}-{interval[2]}")

        key = (chrom, ii)
        ret[key] = stats
    return ret


def get_stats_within(
    sites, 
    interval,
    genotype_dict, 
    map_func, 
    bins,
    mut_map, 
    u_bar,
    get_cross_pop=True,
    phased=False
):
    """
    """
    start, end = interval
    where = np.where((sites >= start) & (sites < end))[0]
    sub_genotype_dict = {p: genotype_dict[p][where] for p in genotype_dict}
    rec_map = map_func(sites[where])

    if mut_map is not None:
        mut_map = mut_map[where]
    else:
        mut_map = None

    sums = compute_stats_within(
        sub_genotype_dict, 
        rec_map, 
        bins,
        mut_map=mut_map,
        u_bar=u_bar,
        get_cross_pop=get_cross_pop,
        phased=phased
    )
    return sums


def get_stats_between(    
    sites, 
    intervals,
    genotype_dict, 
    map_func, 
    bins, 
    mut_map=None, 
    u_bar=None,
    get_cross_pop=True,
    phased=False
):
    """
    Higher-level than `compute_stats_within`. Subsets loaded data 
    """
    (left_start, left_end), (right_start, right_end) = intervals
    where_left = np.where((sites >= left_start) & (sites < left_end))[0]
    left_genotype_dict = {
        p: genotype_dict[p][where_left] for p in genotype_dict}
    left_rec_map = map_func(sites[where_left])

    where_right = np.where((sites >= right_start) & (sites < right_end))[0]
    right_genotype_dict = {
        p: genotype_dict[p][where_right] for p in genotype_dict}
    right_rec_map = map_func(sites[where_right])

    if mut_map is not None:
        left_mut_map = mut_map[where_left]
        right_mut_map = mut_map[where_right]
    else:
        left_mut_map = right_mut_map = None

    sums = compute_stats_between(
        left_genotype_dict,
        right_genotype_dict,
        left_rec_map,
        right_rec_map,
        bins, 
        left_mut_map=left_mut_map,
        right_mut_map=right_mut_map,
        u_bar=u_bar,
        get_cross_pop=get_cross_pop,
        phased=phased
    )
    return sums


def compute_stats_within(
    genotype_dict, 
    rec_map, 
    bins,
    mut_map=None,
    u_bar=None,
    get_cross_pop=True,
    phased=False
):
    """
    
    """
    pop_ids = list(genotype_dict.keys())
    num_pops = len(pop_ids)
    if get_cross_pop:
        num_stats = (num_pops + num_pops ** 2) // 2
    else:
        num_stats = num_pops
    sums = np.zeros((len(bins), num_stats))
    idx = 0
    for ii, pop_i in enumerate(pop_ids):
        for pop_j in pop_ids[ii:]:
            if pop_i == pop_j:
                Gt_ii = genotype_dict[pop_i]
                sums[:-1, idx] = unphased_one_pop_within(
                    Gt_ii, rec_map, bins, mut_map=mut_map, u_bar=u_bar)
            else:
                if not get_cross_pop:
                    continue
                Gt_ii = genotype_dict[pop_i]
                Gt_jj = genotype_dict[pop_j]
                if phased: 
                    pass
                else:
                    sums[:-1, idx] = unphased_cross_pop_within(
                        Gt_ii, Gt_jj, rec_map, bins, 
                        mut_map=mut_map, u_bar=u_bar)
            idx += 1
    sums[-1] = compute_pi(genotype_dict, get_cross_pop=get_cross_pop)
    return sums


def compute_stats_between(
    left_genotype_dict,
    right_genotype_dict,
    left_rec_map,
    right_rec_map,
    bins, 
    left_mut_map=None,
    right_mut_map=None,
    u_bar=None,
    get_cross_pop=True,
    phased=False
):
    """
    
    """
    pop_ids = list(left_genotype_dict.keys())
    num_pops = len(pop_ids)
    if get_cross_pop:
        num_stats = (num_pops + num_pops ** 2) // 2
    else:
        num_stats = num_pops
    sums = np.zeros((len(bins), num_stats))
    idx = 0
    for ii, pop_i in enumerate(pop_ids):
        for pop_j in pop_ids[ii:]:
            if pop_i == pop_j:
                left_Gt_i = left_genotype_dict[pop_i]
                right_Gt_i = right_genotype_dict[pop_i]
                sums[:-1, idx] = unphased_one_pop_between(
                    left_Gt_i, right_Gt_i, left_rec_map, right_rec_map, bins, 
                    left_mut_map=left_mut_map, right_mut_map=right_mut_map, 
                    u_bar=u_bar)
            else:
                if not get_cross_pop:
                    continue
                left_Gt_i = left_genotype_dict[pop_i]
                left_Gt_j = left_genotype_dict[pop_j]
                right_Gt_i = right_genotype_dict[pop_i]
                right_Gt_j = right_genotype_dict[pop_j]
                if phased:
                    pass 
                else:
                    sums[:-1, idx] = unphased_cross_pop_between(
                        left_Gt_i, left_Gt_j, right_Gt_i, right_Gt_j, 
                        left_rec_map, right_rec_map, bins,
                        left_mut_map=left_mut_map, 
                        right_mut_map=right_mut_map, u_bar=u_bar)
            idx += 1
    sums[-1] = 0
    return sums


def unphased_one_pop_within(genotypes, rec_map, bins, mut_map=None, u_bar=None):
    """
    Compute the numerator of the u-adjusted statistic. This is

    u_bar ** 2 * sum_(i,j) D+(i,j) / (u_i * u_j),

    and the whole estimator is

    u_bar ** 2 / n_pairs * sum_(i,j) D+(i,j) / (u_i * u_j)

    :param genotypes: Array of genotypes for a single diploid
    :param rec_map: Array of recombination map coordinates for genotyped sites
    :param bins: Array of recombination bin edges
    :param mut_map: Array of estimated mutation rates at genotyped sites
    """
    # indicator for one-locus heterozygosity
    weights = 1.0 * (genotypes[:, 0] != genotypes[:, 1])
    if u_bar is not None and mut_map is not None:
        assert len(mut_map) == len(rec_map)
        weights *= (u_bar / mut_map)
    stats = _count_locus_pairs(rec_map, bins, weights=weights, verbose=False)
    return stats


def unphased_one_pop_between(
    left_genotypes,
    right_genotypes,
    left_rec_map,
    right_rec_map,
    bins, 
    left_mut_map=None, 
    right_mut_map=None,
    u_bar=None
):
    """
    Compute the numerator of the u-adjusted statistic between two genomic
    intervals. This is

    u_bar ** 2 * sum_(i,j) D+(i,j) / (u_i * u_j),

    and the whole estimator is

    u_bar ** 2 / n_pairs * sum_(i,j) D+(i,j) / (u_i * u_j)

    :param genotypes: Array of genotypes for a single diploid
    :param rec_map: Array of recombination map coordinates for genotyped sites
    :param bins: Array of recombination bin edges
    :param mut_map: Array of estimated mutation rates at genotyped sites
    """
    # indicators for one-locus heterozygosity
    left_weights = 1.0 * (left_genotypes[:, 0] != left_genotypes[:, 1])
    right_weights = 1.0 * (right_genotypes[:, 0] != right_genotypes[:, 1])    
    if u_bar is not None and left_mut_map is not None:
        assert len(left_mut_map) == len(left_rec_map)
        assert len(right_mut_map) == len(right_rec_map)
        left_weights *= (u_bar / left_mut_map)
        right_weights *= (u_bar / right_mut_map)
    stats = _count_locus_pairs_between(left_rec_map, right_rec_map, bins,
        left_weights=left_weights, right_weights=right_weights, verbose=False)
    return stats


def unphased_cross_pop_within(
    genotypes_0, 
    genotypes_1,
    rec_map,
    bins,
    mut_map=None,
    u_bar=None
):
    """
    
    """
    weights = _compute_pi_xy(genotypes_0, genotypes_1)
    if u_bar is not None and mut_map is not None:
        assert len(mut_map) == len(rec_map)
        weights *= (u_bar / mut_map)
    stats = _count_locus_pairs(rec_map, bins, weights=weights, verbose=False)
    return stats


def unphased_cross_pop_between(
    left_genotypes_0, 
    left_genotypes_1,
    right_genotypes_0,
    right_genotypes_1,
    left_rec_map,
    right_rec_map,
    bins,
    left_mut_map=None,
    right_mut_map=None,
    u_bar=None,
):
    """
    
    """
    left_weights = _compute_pi_xy(left_genotypes_0, left_genotypes_1)
    right_weights = _compute_pi_xy(right_genotypes_0, right_genotypes_1)
    if u_bar is not None and left_mut_map is not None:
        assert len(left_mut_map) == len(left_rec_map)
        assert len(right_mut_map) == len(right_rec_map)
        left_weights *= (u_bar / left_mut_map)
        right_weights *= (u_bar / right_mut_map)
    stats = _count_locus_pairs_between(left_rec_map, right_rec_map, bins,
        left_weights=left_weights, right_weights=right_weights, verbose=False)
    return stats


def _phased_cross_pop_within(
    haplotypes_0,
    haplotypes_1,
    rec_map,
    bins,
    mut_map=None,
    u_bar=None
):  
    """
    Evaluate the phased cross-population D+ estimator within a genomic interval.

    Calls itself recursively and returns the average across calls when there 
    are more than one haplotypes in arrays `haplotype_i` and `haplotype_j`. 

    :param haplotypes_i: Haplotype array for sample i
    :param haplotypes_j: Array for sample j
    :param rec_map: Array of recombination map coordinates
    :param bins: Array of recombination bin edges
    :param mut_map: Optional (default None) mutation map for site weighting
    :param u_bar: Optional (default None) parameter for normalizing the 
        mutation map. Required if `mut_map` is given
    
    :returns: Array; binned sums of estimated D+
    """
    n_i = haplotypes_0.shape[1]
    n_j = haplotypes_1.shape[1]
    if n_i == 1 and n_j == 1:
        weights = haplotypes_0[:, 0] != haplotypes_1[:, 0]
        if u_bar is not None and mut_map is not None:
            assert len(mut_map) == len(rec_map)
            weights *= (u_bar / mut_map)
        stats = _count_locus_pairs(
            rec_map, bins, weights=weights, verbose=False)
    else:
        # Average over haplotype-by-haplotype comparisons
        numer = 0.0
        for kk in range(n_i):
            for ll in range(n_j):
                numer += _phased_cross_pop_within(
                    haplotypes_0[:, [kk]], haplotypes_1[:, [ll]], rec_map, 
                    bins, mut_map=mut_map, u_bar=u_bar)
        stats = numer / (n_i * n_j)
    return stats


def _phased_cross_pop_between(
    left_haplotypes_0,
    left_haplotypes_1,
    right_haplotypes_0,
    right_haplotypes_1,
    left_rec_map,
    right_rec_map,
    bins,
    left_mut_map=None,
    right_mut_map=None,
    u_bar=None
):
    """
    Evaluate the phased cross-populatipon D+ estimator between two genomic 
    intervals. 

    See `_phased_cross_pop_within()` for further documentation. 
    """
    n_i = left_haplotypes_0.shape[1]
    assert right_haplotypes_0.shape[1] == n_i
    n_j = left_haplotypes_1.shape[1]
    assert right_haplotypes_1.shape[1] == n_j
    if n_i == 1 and n_j == 1:
        left_weights = left_haplotypes_0[:, 0] != left_haplotypes_1[:, 1]
        right_weights = right_haplotypes_0[:, 0] != right_haplotypes_1[:, 1]
        if u_bar is not None and left_mut_map is not None:
            assert len(left_mut_map) == len(left_rec_map)
            assert len(right_mut_map) == len(right_rec_map)
            left_weights *= (u_bar / left_mut_map)
            right_weights *= (u_bar / right_mut_map)
        stats = _count_locus_pairs_between(
            left_rec_map, right_rec_map, bins, left_weights=left_weights, 
            right_weights=right_weights, verbose=False)
    else:
        numer = 0.0
        for kk in range(n_i):
            for ll in range(n_j):
                numer += _phased_cross_pop_between(
                    left_haplotypes_0[:, [kk]], left_haplotypes_1[:, [ll]],
                    right_haplotypes_0[:, [kk]], right_haplotypes_1[:, [ll]],
                    left_rec_map, right_rec_map, bins, u_bar=u_bar,
                    left_mut_map=left_mut_map, right_mut_map=right_mut_map)
        stats = numer / (n_i * n_j)
    return stats


def get_genotype_dict(genotypes, sample_ids, sample_labels=None):
    """
    sample_labels[label] = sample_id
    """
    if sample_labels is None:
        sample_mapping = {sample_id: sample_id for sample_id in sample_ids}
    else:
        sample_mapping = sample_labels
    n_samples = genotypes.shape[1]
    assert n_samples == len(sample_ids)
    genotype_dict = dict()
    for sample_label in sample_mapping:
        sample_id = sample_mapping[sample_label]
        idx = sample_ids.index(sample_id)
        genotype_dict[sample_label] = genotypes[:, idx]
    return genotype_dict 


def load_label_file(pop_file):
    """
    Load a population file.
    """
    sample_labels = dict()
    with open(pop_file, 'r') as fin:
        for line in fin:
            sample_id, label = line.split()
            if label in sample_labels:
                raise ValueError("Repeated labels in label file")
            sample_labels[label] = sample_id
    return sample_labels


def denoms_within(positions, map_fxn, bins, interval):
    """
    Subset positions to an interval and use `map_fxn` to compute the binned
    D+ denominator with them.

    :param array positions: Array of callable positions
    :param function map_fxn: Function for computing map coordinates from
        `positions`
    :param array bins: Array of bin edges
    :param array interval: Start and end positions of the interval of concern.

    :returns array: Binned locus pair counts
    """
    start, end = interval
    where = np.where((positions >= start) & (positions < end))[0]
    pos_map = map_fxn(positions[where])
    denoms = _count_locus_pairs(pos_map, bins)
    denoms = np.append(denoms, len(where))
    return denoms


def denoms_between(positions, map_fxn, bins, intervals):
    """
    Subset to two intervals and compute binned denominators between them.

    :param array positions: Array of callable positions
    :param function map_fxn: Function for computing map coordinates from
        `positions`
    :param array bins: Array of bin edges
    :param tuple intervals: Nonoverlapping intervals (arrays, length 2) defining 
        lower and upper bounds on left and right loci

    :returns array: Binned locus pair counts
    """
    (lstart, lend), (rstart, rend) = intervals
    where_left = np.where((positions >= lstart) & (positions < lend))[0]
    left_map = map_fxn(positions[where_left])
    where_right = np.where((positions >= rstart) & (positions < rend))[0]
    right_map = map_fxn(positions[where_right])
    denoms = _count_locus_pairs_between(left_map, right_map, bins)
    denoms = np.append(denoms, 0)
    return denoms


def compute_pi(genotype_dict, get_cross_pop=True):
    """
    Compute nucleotide diversity in a contiguous genomic block. Returns an 
    array of sums (to be normalized by L).

    :param pop_genotypes: Dictionary that maps population IDs to arrays of 
        allelic states, generated by `_buld_pop_genotypes`.
    :param cross_pop: If True (default), compute cross-population ``H``
        statistics as well as one-population ones.

    :returns: Array of ``H`` sums. 
    """
    pop_ids = list(genotype_dict.keys())
    num_pops = len(pop_ids)
    if get_cross_pop:
        num_stats = (num_pops + num_pops ** 2) // 2
    else:
        num_stats = num_pops
    sums = np.zeros(num_stats, dtype=np.float64)
    idx = 0
    for i, pop_i in enumerate(pop_ids):
        for pop_j in pop_ids[i:]:
            if pop_i == pop_j: 
                alleles = genotype_dict[pop_i]
                _, n = alleles.shape
                numer = 0.0
                for k in range(n - 1):
                    for l in range(k + 1, n):
                        numer += (alleles[:, k] != alleles[:, l]).sum()
                sum_i = numer / (n * (n - 1) / 2)
                sums[idx] = sum_i
            else:
                if not get_cross_pop:
                    continue
                alleles_i = genotype_dict[pop_i]
                alleles_j = genotype_dict[pop_j]
                _, ni = alleles_i.shape
                _, nj = alleles_j.shape
                numer = 0.0
                for k in range(ni):
                    for l in range(nj):
                        numer += (alleles_i[:, k] != alleles_j[:, l]).sum()
                sum_ij = numer / (ni * nj)
                sums[idx] = sum_ij 
            idx += 1
    return sums


def _compute_pi_xy(genotypes_i, genotypes_j):
    """
    Compute the pairwise divergence between two diploids. This is the nucleotide 
    diversity, conditional on sampling one allele copy from each diploid.

    :param genotypes_i: Array of allelic states with shape (s, 2) for diploid i.
    :param genotypes_j: Array of allelic states for diploid j with shape (s, 2)

    :returns array: Array of site-wise divergences
    """
    pairwise_diff = genotypes_i[:, :, np.newaxis] != genotypes_j[:, np.newaxis]
    pi = pairwise_diff.sum((2, 1)) / 4
    return pi


def _count_locus_pairs(site_map, bins, weights=None, verbose=False):
    """
    Compute the numbers of site pairs that fall within each of a series of 
    recombination bins, in a contiguous genomic window. 

    Used to compute ``D+`` and its denominator. 

    :param site_map: Array giving the recombination map coordinates of sites
        in linear units (cM or M).
    :param bins: Array of recombination bin edges, given in the same unit as 
        the map (cM or M). 
    :weights: An array with length equal to `sitemap` assigning a weight to each 
        site (default None). Computing counts without weights is equivalent to 
        giving every site weight 1.

    :returns: Array of binned locus pair counts.
    """
    num_bins = len(bins) - 1
    sums = np.zeros(num_bins, dtype=np.float64)

    if len(site_map) == 0:
        print(dpluspy._current_time(), 'Empty window: returning 0')
        return sums
    if weights is not None:
        if len(weights) != len(site_map):
            raise ValueError('Length mismatch between `site_map` and `weights`')
    if not np.all(np.diff(site_map) >= 0):
        raise ValueError('`site_map` must increase monotonically')

    if weights is not None:
        if bins[0] == 0:
            indices = np.arange(1, len(site_map) + 1)
        else:
            indices = np.searchsorted(site_map, site_map + bins[0])
        cum_weights = np.concatenate(([0], np.cumsum(weights)))
        cum_sum0 = cum_weights[indices]
        for i, b in enumerate(bins[1:]):
            indices = np.searchsorted(site_map, site_map + b)
            cum_sum1 = cum_weights[indices]
            sums[i] = (weights * (cum_sum1 - cum_sum0)).sum()
            cum_sum0 = cum_sum1
            if verbose:
                print(_current_time(), 
                    f"locus pairs summed (within) in bin {i}")
    else:
        if bins[0] == 0:
            edge0 = np.arange(1, len(site_map) + 1)
        else:
            edge0 = np.searchsorted(site_map, site_map + bins[0])
            assert np.all(edge0 > 0)
        for i, b in enumerate(bins[1:]):
            edge1 = np.searchsorted(site_map, site_map + b)
            sums[i] = (edge1 - edge0).sum() 
            edge0 = edge1
            if verbose:
                print(_current_time(), 
                    f"locus pairs summed (within) in bin {i}")
    return sums


def _count_locus_pairs_between(
    left_map, 
    right_map, 
    bins, 
    left_weights=None, 
    right_weights=None,
    verbose=False
):
    """
    Compute binned counts of locus pairs between two discontinuous genomic
    windows. Used to compute D+ and its denominator. 

    :returns array: Array of binned locus pair counts.
    """
    num_bins = len(bins) - 1
    sums = np.zeros(num_bins, dtype=np.float64)

    if len(left_map) == 0 or len(right_map) == 0:
        print(_current_time(), 'Empty windows: returning 0')
        return sums
    if not np.all(np.diff(left_map) >= 0):
        raise ValueError('`left_map` must increase monotonically')
    if not np.all(np.diff(right_map) >= 0):
        raise ValueError('`right_map` must increase monotonically')
    
    if left_map[-1] > right_map[0]:
        raise ValueError(
            '`right_map` must have higher coords than `left_map`')
    if (left_weights is not None) ^ (right_weights is not None):
        raise ValueError('You must provide weights for both windows')
    if left_weights is not None:
        if len(left_weights) != len(left_map):
            raise ValueError("Map and weight lengths mismatch for block 1")
        if len(right_weights) != len(right_map):
            raise ValueError("Map and weight lengths mismatch for block 2")

    num_bins = len(bins) - 1

    if left_weights is not None:
        indices = np.searchsorted(right_map, left_map + bins[0])
        assert np.all(indices >= 0)
        cum_weights2 = np.concatenate(([0], np.cumsum(right_weights)))
        cum_sum0 = cum_weights2[indices]
        for i, b in enumerate(bins[1:]):
            indices = np.searchsorted(right_map, left_map + b)
            assert np.all(indices >= 0)
            cum_sum1 = cum_weights2[indices]
            sums[i] = (left_weights * (cum_sum1 - cum_sum0)).sum()
            cum_sum0 = cum_sum1
            if verbose:
                print(_current_time(), 
                    f"locus pairs summed (between) in bin {i}")
    else:
        edge0 = np.searchsorted(right_map, left_map + bins[0])
        for i, b in enumerate(bins[1:]):
            edge1 = np.searchsorted(right_map, left_map + b)
            sums[i] = (edge1 - edge0).sum() 
            edge0 = edge1
            if verbose:
                print(_current_time(), 
                    f"locus pairs summed (between) in bin {i}")
    return sums


def _get_uniform_recombination_map(r, L):
    """
    Generate a function that interpolates map coordinates for a uniform 
    recombination with rate `r` and length `L`. 

    :param r: Map rate, in units of r (recombination frequency).
    :param L: Length of the map.

    :returns: Function that interpolates for a uniform map, in M.
    :rtype: scipy.interpolate.interp1d 
    """
    coords = np.arange(1, L + 1)
    map_coords = dpluspy.utils._map_function(r) * np.arange(L)
    map_fxn = scipy.interpolate.interp1d(
        coords, 
        map_coords, 
        kind='nearest', 
        bounds_error=False, 
        fill_value=(map_coords[0], map_coords[-1])
    )
    return map_fxn


def _load_recombination_map(
    filename, 
    pos_col="Position(bp)",
    map_col="Map(cM)",
    interp_method="linear", 
    unit='cM',
    map_sep=None,
    inverse=False
):
    """
    Load a recombination map and return a function that interpolates map 
    positions for sites. Works for maps saved as BEDGRAPH files or in the 
    Hapmap format. The returned map should be in units of Morgans.

    :param str filename: Filename of recombination map.
    :param map_col: Title of column containing map coordinates.
    :param pos_col: Name of position column for hapmap-format files. Default    
        None uses "Position(bp)"
    :param kind: The type of interpolation to use (default 'linear').
    :param unit: The map unit expected in the file (default 'cM'). Values not
        in ('cM', 'M') will raise errors. If 'cM', coordinates are transformed
        to `M`.
    :param sep: If a BEDGRAPH file is given, gives the separator to expect in
        the file (default None uses whitespace).
    :param inverse: If True, return a function that maps from map coordinates
        back to physical coordinates (default False).
        
    :returns: Interpolate function
    :rtype: scipy.interpolate.interp1d 
    """
    if pos_col is None:
        pos_col = "Position(bp)"
    if map_col is None: 
        map_col = "Map(cM)"
    if ".txt" in filename:
        # TODO ADD map_sep
        coords, map_coords = dpluspy.utils._read_hapmap_map(
            filename, map_col=map_col, pos_col=pos_col)
    elif ".bed" or ".bedgraph" in filename:
        coords, map_coords = dpluspy.utils._read_bedgraph_map(
            filename, map_col=map_col, sep=map_sep)
    else:
        try:
            coords, map_coords = dpluspy.utils._read_hapmap_map(
                filename, map_col=map_col, pos_col=pos_col)
        except:
            raise ValueError("Unrecognized recombination map file format")
    if unit not in ('cM', 'M'):
        raise ValueError('Unrecognized map unit')
    if np.any(coords) < 1:
        raise ValueError('All physical coordinates must be greater than 1')
    if unit == 'cM':
        map_coords *= 0.01
    if inverse:
        xs = map_coords
        ys = coords
    else:
        xs = coords
        ys = map_coords
    map_fxn = scipy.interpolate.interp1d(
        xs, 
        ys, 
        kind=interp_method,
        bounds_error=False,
        fill_value=(ys[0], ys[-1]))
    return map_fxn


def _load_mutation_map(filename, positions, map_col="mut_map"):
    """
    Load a mutation map in BEDGRAPH format, or from a site-resolution .npy
    file.

    :param filename: Pathname of the mutation map file.
    :param positions: Array of 1-indexed positions for which to load rates.

    :returns: Site-resolution mutation map array.
    """
    if ".bedgraph" in filename or ".csv" in filename:
        if ".csv" in filename: 
            data = pandas.read_csv(filename, sep=",")
        else:
            data = pandas.read_csv(filename, sep=r"\s+")
        coords = np.array(data["chromEnd"])
        tot_map = np.array(data[map_col])
        if np.any(positions > coords[-1]):
            raise ValueError('Positions exceed map length')
        idxs = np.searchsorted(coords, positions)
        mut_map = tot_map[idxs]
        assert not np.any(np.isnan(mut_map))
    elif filename.endswith('.npy'):
        tot_map = np.load(filename).astype(np.float64)
        if np.any(positions > len(tot_map)):
            raise ValueError('Positions exceed map length')
        mut_map = tot_map[positions - 1]
        assert not np.any(np.isnan(mut_map))
    else:
        raise ValueError('Unrecognized file format')
    return mut_map


def get_vcf_genotypes(
    vcf_file, 
    sample_ids=None,
    bed_file=None, 
    allow_multi=True,
    missing_to_ref=False,
    apply_filter=False,
    interval=None,
    verbose=0
):
    """
    Read sites and genotypes from a VCF file.

    Genotypes are represented in a numpy array with shape `(l, n, 2)`, where 
    `l` is the number of sites and `n` the number of diploid samples. 
    
    :param str vcf_file: Pathname of a VCF file
    :param str bed_file: Optional pathname of BED mask to impose on sites
    :param bool allow_mutli: If True (default), do not skip multiallelic sites
    :param bool missing_to_ref: If True, genotypes ./. and .|. will be read as
        0/0 or 0|0 respectively (default False skips sites with missing data).
    :param tuple interval: Optional 2-tuple/list specifying 1-indexed upper and 
        lower bounds on POS, where the upper bound is noninclusive
    :param verbose: If > 0, print a progress message every `verbose` lines.

    :returns: Array of 1-indexed sites, array of genotypes, list of sample IDs
    """
    if bed_file is not None:
        regions, _ = dpluspy.utils._read_bed_file(bed_file)
        mask = dpluspy.utils._regions_to_mask(regions)
    else:
        mask = None

    if vcf_file.endswith(".gz"):
        opener = gzip.open 
    else:
        opener = open

    with opener(vcf_file, 'rb') as fin:
        sites, genotypes, sample_ids = _read_vcf(
            fin,
            mask=mask,
            sample_ids=sample_ids,
            allow_multi=allow_multi,
            missing_to_ref=missing_to_ref,
            apply_filter=apply_filter,
            interval=interval,
            verbose=verbose
        )
    return sites, genotypes, sample_ids


def get_ts_genotypes(
    ts, 
    ts_sample_ids=None,
    sample_ids=None,
    bed_file=None, 
    allow_multi=True,
    missing_to_ref=False,
    apply_filter=False,
    interval=None,
    verbose=0
):
    """
    Read an array of sites and an array of genotype codes from a tskit tree
    sequence with mutations. 

    :param TreeSequence ts: Tskit tree sequence
    :param list ts_sample_ids: Optional IDs for tree sequence samples; if not
        given then ts samples are named "tsk0", "tsk1", etc.
    :param list sample_ids: Optional list of sample IDs to include
    :param str bed_file: Optional pathname of BED mask file
    :param bool allow_multi: If True (default), allow multiallelic sites
    :param bool missing_to_ref: If True (default False), sets missing alleles 
        to "0"; otherwise skips sites with missing data
    :param bool apply_filter: If True, skips sites with "FILTER" column not
        equal to "PASS" or "."
    :param tuple interval: Optional interval for sites
    :param int verbose: If > 0, prints progress messages every `verbose` lines

    :returns tuple: Sites array, genotype array, list of sample IDs
    """
    if bed_file is not None:
        regions, _ = dpluspy.utils._read_bed_file(bed_file)
        mask = dpluspy.utils._regions_to_mask(regions)
    else: 
        mask = None

    vcf_str = ts.as_vcf(position_transform=dpluspy.utils._increment1, 
        individual_names=ts_sample_ids)

    with io.StringIO(vcf_str) as fin:
        sites, genotypes, sample_ids = _read_vcf(
            fin,
            mask=mask,
            sample_ids=sample_ids,
            allow_multi=allow_multi,
            missing_to_ref=missing_to_ref,
            apply_filter=apply_filter,
            interval=interval,
            verbose=verbose
        )
    return sites, genotypes, sample_ids


def _read_vcf(
    fin, 
    sample_ids=None,
    mask=None, 
    allow_multi=True,
    missing_to_ref=False,
    apply_filter=False,
    interval=None,
    verbose=0
):
    """
    Read sites, an array of genotype codes, and sample IDs from an opened VCF 
    file or file-like object. 

    :param fin: 
    :param list sample_ids: Optional list of sample IDs to include in output
    :param array mask: Optional site-resolution genetic mask array. Should
        equal True where sites are excluded by the mask.
    :param bool allow_multi: If True (default), allow multiallelic sites
    :param bool missing_to_ref: If True (default False), sets missing alleles 
        to "0"; otherwise skips sites with missing data
    :param bool apply_filter: If True, skips sites with "FILTER" column not
        equal to "PASS" or "."
    :param tuple interval: Optional interval for sites
    :param int verbose: If > 0, prints progress messages every `verbose` lines

    :returns tuple: Sites array, genotype array, list of sample IDs
    """
    sites = list() 
    genotypes = list()
    counter = 0

    for line in fin:
        if isinstance(line, bytes):
            line = line.decode()
        if line.startswith('#'):
            if line.startswith('#CHROM'):
                all_ids = line.split()[9:]
                if sample_ids is None:
                    sample_ids = all_ids 
                    sample_idx = list(range(len(all_ids)))
                else:
                    sample_idx = [all_ids.index(x) for x in sample_ids]
            continue

        split_line = line.split()
        pos1 = int(split_line[1])
        if verbose > 1:
            if counter % verbose == 0 and counter > 1:
                print(_current_time(),
                    f'parsed POS {pos1} line {counter}')
        counter += 1

        # Filtering on the site
        if interval is not None:
            if pos1 < interval[0]:
                continue
            if pos1 >= interval[1]:
                break
        if mask is not None:
            pos0 = pos1 - 1
            if pos0 >= len(mask):
                break
            if mask[pos0] == True:
                continue
        if apply_filter:
            filtr = split_line[6]
            if filtr not in ('PASS', '.'):
                continue
        
        ref = split_line[3]
        alts = split_line[4].split(',')
        alleles = [ref] + alts

        # Filter non-SNVs, and multiallelic sites if `allow_multi` is False
        if np.any([len(allele) > 1 for allele in alleles]):
            continue
        if not allow_multi:
            if len(alts) > 1:
                continue

        samples = [split_line[9:][idx] for idx in sample_idx]
        split_samples = [sample.split(':') for sample in samples]
        genotype_strs = [sample[0] for sample in split_samples]
        genotype_list = [re.split("/|\\|", gt) for gt in genotype_strs]
        skip_line = False
        for i, gt in enumerate(genotype_list):
            for j, allele in enumerate(gt):
                if allele == '.':
                    if missing_to_ref:
                        genotype_list[i][j] = '0'
                    else:
                        warnings.warn(f"Missing genotype at site {pos1}")
                        skip_line = True
        if skip_line:
            continue
        genotypes.append(np.array(genotype_list))
        sites.append(pos1)

    sites = np.array(sites, np.int64)
    genotypes = np.array(genotypes, np.int64)
    return sites, genotypes, sample_ids


def _current_time():
    """
    Return a string giving the time and date with yyyy-mm-dd format.
    """
    return "[" + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + "]"

