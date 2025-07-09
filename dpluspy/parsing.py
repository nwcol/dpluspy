"""
Houses functions for estimating D+ and one-locus diversity from sequence data.
"""

from collections import defaultdict
import io
import gzip
import numpy as np
import pandas
import re
import scipy
import warnings

from . import utils


def parse_stats(
    vcf_file,
    ts_sample_ids=None,
    bed_file=None,
    pop_file=None,
    pop_mapping=None,
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
    interval=None,
    intervals=None,
    interval_file=None,
    chrom="None",
    phased=False,
    get_cross_pop=True,
    get_denoms=True,
    allow_multi=True,
    missing_to_ref=False,
    apply_filter=False,
    overhang="merge",
    verbose=True
):
    """
    Compute D+, H and their denominators from a VCF file or Tskit tree sequence
    in (and between) one or more genomic intervals.
    
    :param str vcf_file: Pathname of VCF file, or optionally a Tskit tree
        sequence instance with mutations. Provides sites and genotypes.
    :param list ts_sample_ids: Names of ts samples when a tree seq is given.
    :param str bed_file: Pathname of BED file defining regions to parse,
        required to compute the denominator
    :param str pop_file: Optional pathname of whitespace-separated file mapping 
        sample IDs to populations. Default behavior takes each VCF sample as a 
        member of a distinct population
    :param dict pop_mapping: Optional dictionary mapping populations to lists 
        of sample IDs
    :param str rec_map_file: Optional recombination map in HapMap or BEDGRAPH
        format. `rec_map_file` or `r` must be given.
    :param str pos_col: Rec. map "position" column (default "Position(bp)")
    :param str pos_col: Rec. map "map" column to use (default "Map(cM)")
    :param str map_sep: 
    :param str interp_method: Method for interpolating rec. map coordinates
    :param float r: Uniform recombination rate for map interpolation, primarily
        for use when parsing simulated data
    :param array r_bins: Bin edges given in recombination fraction units.
    :param array bp_bins: Bin edges in physical units (base pairs).
    :param str mut_map_file: Pathname of a BEDGRAPH or site-resolution .npy 
        file containing estimated mutation rates. If provided, `mut_facs` will
        be computed and returned with other statistics for use in weighting.
    :param str mut_col:
    :param array interval: Single genomic interval to parse
    :param list intervals: List of genomic intervals to parse
    :param str interval_file: Pathname of whitespace-separated file holding
        one interval on each line
    :param str chrom: Optional chromosome ID, used to name intervals
    :param bool phased: If True, treat all VCF data as phased and use the 
        haplotype estimators for D+ (default False)
    :param bool get_cross_pop: If True (default), compute cross-population 
        D+ and H statistics
    :param bool get_denoms: If true (default), compute the denominator for D+.
    :param bool allow_multi: If True (default), parse over multiallelic sites
    :param bool missing_to_ref: If True (default False), convert missing allele
        data to the reference; otherwise sites with missing data are skipped
    :param bool apply_filter: If True, exclude VCF sites with "FAIL" in "FILTER"
    :param str overhang: Method to use for computing/recording D+ statistic 
        between genomic intervals. See `compute_stats` for more information.
    :param bool verbose: If True (default), print reports.

    :returns dict: A dictionary holding raw windowed sums of D+ and H statistics
    """
    if interval is not None:
        intervals = [interval]
    if interval_file is not None:
        intervals = np.loadtxt(interval_file)
    intervals = [np.asarray(x).flatten() for x in intervals]

    if get_denoms:
        if bed_file is not None: 
            positions = utils._read_bed_file_positions(bed_file) + 1
            seq_length = positions[-1]
        else:
            seq_length = intervals[-1][-1]
            positions = np.arange(1, seq_length)

        if mut_map_file is not None:
            mut_map = _load_mutation_map(
                mut_map_file, positions, map_col=mut_col)
        else:
            mut_map = None
    else:
        positions = None
        mut_map = None
        seq_length = intervals[-1][-1] + 1

    if r_bins is not None:
        if isinstance(r_bins, str):
            r_bins = np.loadtxt(r_bins)
        # Convert bins in r to Morgans 
        bins = utils._map_function(r_bins)
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

    if pop_file is not None:
        pop_mapping = _load_pop_file(pop_file)

    if pop_mapping is not None:
        vcf_ids = [sample for pop in pop_mapping for sample in pop_mapping[pop]]
    else:
        vcf_ids = None

    # Read genotypes from a VCF file or extract them from a tree sequence
    if isinstance(vcf_file, str):
        sites, genotypes, sample_ids = get_vcf_genotypes(
            vcf_file, 
            sample_ids=vcf_ids,
            bed_file=bed_file, 
            allow_multi=allow_multi,
            missing_to_ref=missing_to_ref,
            apply_filter=apply_filter
        )
    else:
        sites, genotypes, sample_ids = get_ts_genotypes(
            vcf_file, 
            ts_sample_ids=ts_sample_ids,
            sample_ids=vcf_ids,
            bed_file=bed_file, 
            allow_multi=allow_multi,
            missing_to_ref=missing_to_ref,
            apply_filter=apply_filter
        )
    pop_genotypes = _build_pop_genotypes(
        genotypes, sample_ids, pop_mapping=pop_mapping)
    
    stats = compute_stats(    
        sites,
        pop_genotypes,
        map_fxn,
        bins,
        intervals,
        positions=positions,
        mut_map=mut_map,
        chrom=chrom,
        get_cross_pop=get_cross_pop,
        phased=phased,
        overhang=overhang,
        ret_bins=ret_bins,
        verbose=verbose
    )
    return stats


def compute_stats(
    sites,
    pop_genotypes,
    map_fxn,
    bins,
    intervals,
    positions=None,
    mut_map=None,
    chrom="None",
    get_cross_pop=True,
    phased=False,
    overhang="merge",
    verbose=True,
    ret_bins=None
):
    """
    Compute D+, H statistics and their denominators from loaded data.

    :param array sites: Array of VCF site positions
    :param dict pop_genotypes: Dictionary mapping population names to genotype
        arrays
    :param function map_fxn: Function for computing map coordinates from 
        physical positions
    :param array bins: Bin edges; may be in Morgans or physical units (bp)
    :param list intervals: Window intervals.
    :param array positions: Array of callable positions
    :param array mut_map: Optional mutation map, 
    :param str chrom: Optional chromosome ID used to name intervals 
        (default "None")
    :param bool get_cross_pop: If True (default), compute and return cross-
        population D+ and H.
    :param bool phased: If True (default False), treat data as phased and use
        the haplotype estimators.
    :param str overhang: Determines how locus pairs that span multiple genomic
        intervals are handled. TODO write details
        The third element of each interval provides an upper bound on the 
        positions of right loci.
    :param bool verbose: If True (default), print progress messages as intervals
        are parsed.
    :param array ret_bins: Bins to return as part of output data (for use when
        specifying bins in units of r)

    :returns dict: A dictionary mapping interval names to sums of statistics.
        Each interval dict has keys "sums", "pop_ids", "bins", plus optionally
        "denoms" and "mut_facs".
    """
    if ret_bins is None:
        ret_bins = bins 

    pop_ids = list(pop_genotypes.keys())
    denoms = None
    mut_facs = None 

    ret = dict()

    if overhang == "merge" or overhang == "merged":
        for ii, interval in enumerate(intervals):
            assert len(interval) == 3
            interval0 = interval[:2]
            interval1 = interval[1:]
            _intervals = (interval0, interval1)
            if positions is not None:
                denoms = denoms_within(positions, map_fxn, bins, interval0)
                if interval1[1] > interval1[0]:
                    denoms += denoms_between(
                        positions, map_fxn, bins, _intervals)
            if mut_map is not None:
                mut_facs = mut_facs_within(
                    positions, mut_map, map_fxn, bins, interval0)
                if interval1[1] > interval1[0]:
                    mut_facs += mut_facs_between(
                        positions, mut_map, map_fxn, bins, _intervals)
            sums = stats_within(
                sites, 
                pop_genotypes, 
                map_fxn, 
                bins, 
                interval0, 
                get_cross_pop=get_cross_pop, 
                phased=phased
            )
            if interval1[1] > interval1[0]:
                sums += stats_between(
                    sites, 
                    pop_genotypes, 
                    map_fxn, 
                    bins, 
                    _intervals, 
                    get_cross_pop=get_cross_pop, 
                    phased=phased
                )
            stats = dict()
            stats["bins"] = ret_bins
            stats["pop_ids"] = pop_ids
            stats["sums"] = sums
            if denoms is not None:
                stats["denoms"] = denoms
            if mut_facs is not None:
                stats["mut_facs"] = mut_facs
            key = (chrom, ii)
            ret[key] = stats

            if verbose:
                print(utils._current_time(), 
                    f"Computed stats in chrom {chrom} interval {ii}")
        
    elif overhang == "fancy":
        for ii, left_interval in enumerate(intervals):
            # Parse within interval ii
            assert len(left_interval) == 3
            interval0 = left_interval[:2]
            if positions is not None:
                denoms = denoms_within(positions, map_fxn, bins, interval0)
            if mut_map is not None:
                mut_facs = mut_facs_within(
                    positions, mut_map, map_fxn, bins, interval0)
            sums = stats_within(
                sites, 
                pop_genotypes, 
                map_fxn, 
                bins, 
                interval0, 
                get_cross_pop=get_cross_pop, 
                phased=phased
            )
            stats = dict()
            stats["bins"] = ret_bins
            stats["pop_ids"] = pop_ids
            stats["sums"] = sums
            if denoms is not None:
                stats["denoms"] = denoms
            if mut_facs is not None:
                stats["mut_facs"] = mut_facs
            key = (chrom, (ii, ii))
            ret[key] = stats

            if verbose:
                print(utils._current_time(), 
                    f"Computed stats in chrom {chrom} interval {ii}")

            # Parse between interval ii and accessible intervals to its right
            for jj in range(ii + 1, len(intervals)):
                right_interval = intervals[jj]
                if left_interval[2] < right_interval[1]:
                    continue
                _intervals = (left_interval[:2], right_interval[:2])
                if positions is not None:
                    denoms = denoms_between(
                        positions, map_fxn, bins, _intervals)
                if mut_map is not None:
                    mut_facs = mut_facs_between(
                        positions, mut_map, map_fxn, bins, _intervals)
                sums = stats_between(
                    sites, 
                    pop_genotypes, 
                    map_fxn, 
                    bins, 
                    _intervals, 
                    get_cross_pop=get_cross_pop, 
                    phased=phased
                )
                stats = dict()
                stats["bins"] = ret_bins
                stats["pop_ids"] = pop_ids
                stats["sums"] = sums
                if denoms is not None:
                    stats["denoms"] = denoms
                if mut_facs is not None:
                    stats["mut_facs"] = mut_facs
                key = (chrom, (ii, jj))
                ret[key] = stats

                if verbose:
                    print(utils._current_time(), "Computed stats between "
                        f"chrom {chrom} intervals ({ii}, {jj})")

    else:
        for ii, interval in enumerate(intervals):
            interval0 = interval[:2]
            if positions is not None:
                denoms = denoms_within(positions, map_fxn, bins, interval0)
            if mut_map is not None:
                mut_facs = mut_facs_within(
                    positions, mut_map, map_fxn, bins, interval0)
            sums = stats_within(
                sites, 
                pop_genotypes,
                map_fxn, 
                bins, 
                interval0, 
                get_cross_pop=get_cross_pop, 
                phased=phased
            )
            stats = dict()
            stats["bins"] = ret_bins
            stats["pop_ids"] = pop_ids
            stats["sums"] = sums
            if denoms is not None:
                stats["denoms"] = denoms
            if mut_facs is not None:
                stats["mut_facs"] = mut_facs
            key = (chrom, ii)
            ret[key] = stats

            if verbose:
                print(utils._current_time(), 
                    f"Computed stats in chrom {chrom} interval {ii}")
    
    return ret


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


def mut_facs_within(positions, mut_map, map_fxn, bins, interval):
    """
    Subset mutation data to an interval and compute mutation factors (sums of
    mutation-rate products across locus pairs) within it.

    :param array positions: Array of callable positions
    :param array mut_map: Estimated mutation rates, corresponding to `positions`
    :param function map_fxn: Function for computing recombination map 
        coordinates from `positions`
    :param array bins: Array of distance bin edges
    :param array interval: Upper and lower bounds on locus positions

    :returns array: Mutation factors
    """
    start, end = interval
    where = np.where((positions >= start) & (positions < end))[0]
    sub_mut_map = mut_map[where]
    pos_map = map_fxn(positions[where])
    mut_facs = _count_locus_pairs(pos_map, bins, weights=sub_mut_map)
    sum_mut = np.sum(sub_mut_map)
    mut_facs = np.append(mut_facs, sum_mut)
    return mut_facs


def mut_facs_between(positions, mut_map, map_fxn, bins, intervals):
    """
    Subset mutation data to two intervals and compute mutation factors between
    them.

    :param tuple intervals: Nonoverlapping intervals (arrays, length 2) defining 
        lower and upper bounds on left and right loci

    :returns array: Mutation factors
    """
    (lstart, lend), (rstart, rend) = intervals
    where_left = np.where((positions >= lstart) & (positions < lend))[0]
    left_mut_map = mut_map[where_left]
    left_map = map_fxn(positions[where_left])

    where_right = np.where((positions >= rstart) & (positions < rend))[0]
    right_map = map_fxn(positions[where_right])
    right_mut_map = mut_map[where_right]

    mut_facs = _count_locus_pairs_between(
        left_map, 
        right_map, 
        bins, 
        left_weights=left_mut_map, 
        right_weights=right_mut_map
    )
    mut_facs = np.append(mut_facs, 0)
    return mut_facs


def stats_within(
    sites, 
    pop_genotypes, 
    map_fxn, 
    bins, 
    interval, 
    get_cross_pop=True,
    phased=True
):
    """
    Subset data to an interval and compute statistics within it.

    :param array sites: Sites corresponding to genotypes
    :param dict pop_genotypes: Dictionary mapping population names to arrays
        of sample genotypes
    :param function map_fxn: Function mapping physical positions to map coords
    :param array bins: Distance (physical or recombination) bin edges
    :param array interval: Start and end of interval to work within
    :param bool get_cross_pop: If True (default), compute cross-population 
        statistics and include them in output
    :param bool phased: If True (default False), treat data as phased and use 
        haplotype estimators

    :returns array: Array of sums
    """
    start, end = interval
    where = np.where((sites >= start) & (sites < end))[0]
    sub_genotypes = {p: pop_genotypes[p][where] for p in pop_genotypes}
    site_map = map_fxn(sites[where])
    sums = _compute_stats_within(
        sub_genotypes, 
        site_map, 
        bins, 
        cross_pop=get_cross_pop, 
        phased=phased
    )
    return sums


def stats_between(    
    sites, 
    pop_genotypes, 
    map_fxn, 
    bins, 
    intervals, 
    get_cross_pop=True,
    phased=True
):
    """
    Subset data to two intervals and compute statistics between them. See 
    `stats_within` for parameter definitions.

    :param tuple intervals: Nonoverlapping intervals (arrays, length 2) defining 
        lower and upper bounds on left and right loci

    :returns array: Array of sums
    """
    (lstart, lend), (rstart, rend) = intervals
    where_left = np.where((sites >= lstart) & (sites < lend))[0]
    left_genotypes = {p: pop_genotypes[p][where_left] for p in pop_genotypes}
    left_map = map_fxn(sites[where_left])

    where_right = np.where((sites >= rstart) & (sites < rend))[0]
    right_genotypes = {p: pop_genotypes[p][where_right] for p in pop_genotypes}
    right_map = map_fxn(sites[where_right])

    sums = _compute_stats_between(
        left_genotypes, 
        right_genotypes, 
        left_map, 
        right_map, 
        bins, 
        cross_pop=get_cross_pop, 
        phased=phased
    )
    return sums
    

def _load_pop_file(pop_file):
    """
    Load a population file.

    :param str pop_file: Pathname of population file.
    
    :returns dict: Dictionary mapping population IDs to lists of sample IDs
    """
    pop_mapping = defaultdict(list)
    with open(pop_file, 'r') as fin:
        for line in fin:
            sample, pop = line.split()
            pop_mapping[pop].append(sample)
    return pop_mapping


def _build_pop_genotypes(genotypes, sample_ids, pop_mapping=None):
    """
    Break an array of genotypes corresponding to `sample_ids` up into 
    population-specific arrays stored in a dictionary.
    
    :param array genotypes: Array of allelic states loaded by `read_vcf`. 
        This should have the shape `(l, n, 2)`, where `l` is the number of
        sites and `n` is the number of diploid samples.
    :param list sample_ids: VCF sample IDs loaded by `read_vcf`.
    :param dict pop_mapping: Dictionary mapping population IDs to lists of 
        sample IDs (default None). If None, each sample is placed in a unique
        population.
    
    returns dict: Mapping of population IDs to population-specific genotype 
        arrays
    """
    if pop_mapping is None:
        pop_mapping = {sample_id: [sample_id] for sample_id in sample_ids}
    pop_ids = list(pop_mapping.keys())
    pop_indices = {}
    for pop_id in pop_ids:
        samples = pop_mapping[pop_id]
        pop_indices[pop_id] = [sample_ids.index(sample) for sample in samples]
    pop_genotypes = {}
    for pop_id in pop_ids:
        pop_genotypes[pop_id] = genotypes[:, pop_indices[pop_id]]
    return pop_genotypes 


def _flatten_pop_genotypes(pop_genotypes):
    """
    Convert a dictionary of arrays with shapes ``(s, n, 2)`` to a dictionary of
    arrays with shapes ``(s, 2 * n)``. Here these are interpreted as arrays of 
    haplotypes or haploid genomes, which are used to estimate the phased ``D+``
    statistic. In some other contexts it is also convenient to have genotypes 
    represented in this way (e.g. estimating pairwise diversity).

    :param pop_genotypes: Dictionary that maps population IDs to arrays of 
        allelic states, generated by `_buld_pop_genotypes`.

    :returns: Dictionary mapping population IDs to arrays of allelic stats that
        have been flattened over the last axis.
    """
    flat_genotypes = {}
    for pop_id in pop_genotypes:
        array = pop_genotypes[pop_id]
        s, n, _ = array.shape
        flat_array = np.reshape(array, (s, 2 * n))
        flat_genotypes[pop_id] = flat_array
    return flat_genotypes


def _compute_pi(pop_genotypes, cross_pop=True):
    """
    Compute nucleotide diversity in a contiguous genomic block. Returns an 
    array of sums (to be normalized by L).

    :param pop_genotypes: Dictionary that maps population IDs to arrays of 
        allelic states, generated by `_buld_pop_genotypes`.
    :param cross_pop: If True (default), compute cross-population ``H``
        statistics as well as one-population ones.

    :returns: Array of ``H`` sums. 
    """
    flat_genotypes = _flatten_pop_genotypes(pop_genotypes)
    pop_ids = list(flat_genotypes.keys())
    num_pops = len(pop_ids)
    if cross_pop:
        num_stats = (num_pops + num_pops ** 2) // 2
    else:
        num_stats = num_pops
    sums = np.zeros(num_stats, dtype=np.float64)
    idx = 0
    for i, pop_i in enumerate(pop_ids):
        for pop_j in pop_ids[i:]:
            if pop_i == pop_j: 
                alleles = flat_genotypes[pop_i]
                _, n = alleles.shape
                numer = 0.0
                for k in range(n - 1):
                    for l in range(k + 1, n):
                        numer += (alleles[:, k] != alleles[:, l]).sum()
                sum_i = numer / (n * (n - 1) / 2)
                sums[idx] = sum_i
            else:
                if not cross_pop:
                    continue
                alleles_i = flat_genotypes[pop_i]
                alleles_j = flat_genotypes[pop_j]
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


def _compute_stats_within(
    pop_genotypes, 
    site_map, 
    bins, 
    cross_pop=True,
    phased=False,
):
    """
    Compute ``D+`` statistic in a contiguous genomic block.

    :param pop_genotypes: Should instead hold haplotypes if `phased` is True.
    :param site_map: Site-wise array of recombination map coordinates.
    :param bins: Recombination distance bins. These should be in the same units
        as `site_map`, which are typically Morgans and could be cM 
        (centiMorgans). 
    :param cross_pop: If True (default), compute cross-population ``D+``
        statistics as well as one-population ones.
    :param phased: If True (default False), use phased (haplotype) estimators
        rather than unphased (genotype) estimators.
    
    :returns: Array of binned ``D+`` sums.
    """
    pop_ids = list(pop_genotypes.keys())
    num_pops = len(pop_ids)
    if cross_pop:
        num_stats = (num_pops + num_pops ** 2) // 2
    else:
        num_stats = num_pops
    sums = np.zeros((len(bins), num_stats))
    idx = 0
    for i, pop_i in enumerate(pop_ids):
        for pop_j in pop_ids[i:]:
            if pop_i == pop_j:
                Gt = pop_genotypes[pop_i]
                if phased:
                    sums[:-1, idx] = _haplotype_D_plus(Gt, site_map, bins)
                else:
                    sums[:-1, idx] = _genotype_D_plus(Gt, site_map, bins)
            else:
                if not cross_pop:
                    continue
                Gi = pop_genotypes[pop_i]
                Gj = pop_genotypes[pop_j]
                if phased:
                    sums[:-1, idx] = _cross_haplotype_D_plus(
                        Gi, Gj, site_map, bins)
                else:
                    sums[:-1, idx] = _cross_genotype_D_plus(
                        Gi, Gj, site_map, bins)
            idx += 1
    sums[-1] = _compute_pi(pop_genotypes, cross_pop=cross_pop)
    return sums


def _compute_stats_between(
    pop_genotypes_left,
    pop_genotypes_right,
    left_mapeft,
    right_mapight,
    bins, 
    cross_pop=True,
    phased=False
):
    """
    Compute ``D+`` statistic between two contiguous genomic blocks.

    :param pop_genotypes_left: Dictionary of population genotype arrays for
        the left window.
    :param pop_genotypes_right: Dictionary of population genotype arrays for
        the right window.
    :param left_mapeft: Site-wise array of recombination map coordinates for
        the left window.
    :param bins: Recombination distance bins. These should be in the same units
        as `site_map`, which are typically Morgans and could be cM.
    :param cross_pop: If True (default), compute cross-population ``D+``
        statistics as well as one-population ones.
    :param phased: If True (default False), use phased (haplotype) estimators
        rather than unphased (genotype) estimators.
    
    :returns: Array of binned ``D+`` sums.
    """
    pop_ids = list(pop_genotypes_left.keys())
    num_pops = len(pop_ids)
    if cross_pop:
        num_stats = (num_pops + num_pops ** 2) // 2
    else:
        num_stats = num_pops
    sums = np.zeros((len(bins), num_stats))
    idx = 0
    for i, pop_i in enumerate(pop_ids):
        for pop_j in pop_ids[i:]:
            if pop_i == pop_j:
                G_l = pop_genotypes_left[pop_i]
                G_r = pop_genotypes_right[pop_i]
                if phased:
                    sums[:-1, idx] = _haplotype_D_plus_between(
                        G_l, G_r, left_mapeft, right_mapight, bins)
                else:
                    sums[:-1, idx] = _genotype_D_plus_between(
                        G_l, G_r, left_mapeft, right_mapight, bins)
            else:
                if not cross_pop:
                    continue
                G_li = pop_genotypes_left[pop_i]
                G_lj = pop_genotypes_left[pop_j]
                G_ri = pop_genotypes_right[pop_i]
                G_rj = pop_genotypes_right[pop_j]
                if phased:
                    sums[:-1, idx] = _cross_haplotype_D_plus_between(
                        G_li, G_lj, G_ri, G_rj, 
                        left_mapeft, right_mapight, bins)
                else:
                    sums[:-1, idx] = _cross_genotype_D_plus_between(
                        G_li, G_lj, G_ri, G_rj, 
                        left_mapeft, right_mapight, bins)
            idx += 1
    sums[-1] = 0
    return sums


def _haplotype_D_plus(haplotypes, site_map, bins):
    """
    The one-population phased estimator for contiguous genomic regions. Calls
    itself recursively and returns a mean across haplotype pairs when there 
    are >2 haplotypes.
    """
    n = haplotypes.shape[1]
    if n == 2:
        weights = haplotypes[:, 0] != haplotypes[:, 1]
        D_plus = _count_locus_pairs(site_map, bins, weights=weights)
    else:
        numer = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                numer += _haplotype_D_plus(
                    haplotypes[:, [i, j]], site_map, bins)
        D_plus = numer / (n * (n - 1) / 2)
    return D_plus


def _haplotype_D_plus_between(
    left_haplotypes, 
    right_haplotypes, 
    left_map, 
    right_map, 
    bins
):
    """
    The one-population phased estimator between two genomic regions. When there
    are >2 haplotypes in the sample, calls itself recursively and returns a 
    mean across n choose 2 haplotype pairs.
    """
    n = left_haplotypes.shape[1]
    if n == 2:
        left_weights = left_haplotypes[:, 0] != left_haplotypes[:, 1]
        right_weights = right_haplotypes[:, 0] != right_haplotypes[:, 1]
        D_plus = _count_locus_pairs_between(
            left_map,
            right_map, 
            bins, 
            left_weights=left_weights,
            right_weights=right_weights
        )
    else:
        numer = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                numer += _haplotype_D_plus_between(
                    left_haplotypes[:, [i]], 
                    right_haplotypes[:, [j]], 
                    left_map,
                    right_map, 
                    bins
                )
        D_plus = numer / (n * (n - 1) / 2)
    return D_plus


def _cross_haplotype_D_plus(haplotypes_i, haplotypes_j, site_map, bins):
    """
    The cross-population phased estimator within a genomic region. When one or
    more samples has >1 haplotypes, calls itself recursively and returns an 
    average across the ``n_i * n_j`` unique haplotype pairs.
    """
    ni = haplotypes_i.shape[1]
    nj = haplotypes_j.shape[1]
    if ni == 1 and nj == 1:
        weights = haplotypes_i[:, 0] != haplotypes_j[:, 0]
        D_plus = _count_locus_pairs(site_map, bins, weights=weights)
    else:
        numer = 0.0
        for k in range(ni):
            for l in range(nj):
                numer += _cross_haplotype_D_plus(
                    haplotypes_i[:, [k]], haplotypes_j[:, [l]], site_map, bins
                )
        D_plus = numer / (ni * nj)
    return D_plus
 

def _cross_haplotype_D_plus_between(
    left_haplotypes_i, 
    left_haplotypes_j,
    right_haplotypes_i, 
    right_haplotypes_j,
    left_map, 
    right_map, 
    bins
):
    """
    The cross-population phased between-genomic-blocks ``D+`` estimator. 
    """
    ni = left_haplotypes_i.shape[1]
    nj = left_haplotypes_j.shape[1]
    if ni == 1 and nj == 1:
        left_weights = left_haplotypes_i[:, 0] != left_haplotypes_j[:, 1]
        right_weights = right_haplotypes_i[:, 0] != right_haplotypes_j[:, 1]
        D_plus = _count_locus_pairs_between(
            left_map,
            right_map, 
            bins, 
            left_weights=left_weights,
            right_weights=right_weights
        )
    else:
        numer = 0.0
        for k in range(ni):
            for l in range(nj):
                numer += _haplotype_D_plus_between(
                    left_haplotypes_i[:, [k]], 
                    left_haplotypes_j[:, [l]],
                    right_haplotypes_i[:, [k]], 
                    right_haplotypes_j[:, [l]],
                    left_map,
                    right_map, 
                    bins
                )
        D_plus = numer / (ni * nj)
    return D_plus


def _genotype_D_plus(genotypes, site_map, bins):
    """
    The one-population unphased ``D+`` estimator, for use within a genomic 
    region. When there are >1 diploids in the sample, returns an average over
    ``n`` within-diploid estimates.

    :param genotypes: Array with shape ``(s, n, 2)``, where ``s`` is the number
        of sites and ``n`` is the number of diploid samples.
    :param site_map: Array of site recombination map coordinates, in M or cM.
    :param bins: Array of recombination bin edges, in M or cM.

    :returns: Array of binned ``D+`` sums.
    """
    n = genotypes.shape[1]
    if n == 1:
        weights = genotypes[:, 0, 0] != genotypes[:, 0, 1]
        D_plus = _count_locus_pairs(site_map, bins, weights=weights)
    else:
        numer = 0.0
        for i in range(n):
            numer += _genotype_D_plus(genotypes[:, [i], :], site_map, bins)
        D_plus = numer / n
    return D_plus


def _genotype_D_plus_between(
    left_genotypes,
    right_genotypes,
    left_map,
    right_map,
    bins
):
    """
    The one-population, unphased, between-genomic-blocks ``D+`` estimator. 
    When there are >1 diploids in the sample, returns an average over ``n`` 
    within-diploid estimates (where ``n`` is a count of diploid samples).
    """
    n = left_genotypes.shape[1]
    if n == 1:
        left_weights = left_genotypes[:, 0, 0] != left_genotypes[:, 0, 1]
        right_weights = right_genotypes[:, 0, 0] != right_genotypes[:, 0, 1]
        D_plus = _count_locus_pairs_between(
            left_map, 
            right_map, 
            bins,
            left_weights=left_weights, 
            right_weights=right_weights
        )
    else:
        numer = 0.0
        for i in range(n):
            numer += _genotype_D_plus_between(
                left_genotypes[:, [i], :], 
                right_genotypes[:, [i], :],
                left_map,
                right_map,
                bins
            )
        D_plus = numer / n
    return D_plus


def _cross_genotype_D_plus(genotypes_i, genotypes_j, site_map, bins):
    """
    The one-population unphased ``D+`` within-block estimator. When there are 
    >1 diploids in one of the populations, returns an average over the 
    ``n_i * n_j`` between-diploid pairs (where ``n`` is a count of diploid
    samples)
    """
    ni = genotypes_i.shape[1]
    nj = genotypes_j.shape[1]
    if ni == 1 and nj == 1:
        weights = _compute_pi_xy(genotypes_i[:, 0], genotypes_j[:, 0])
        D_plus = _count_locus_pairs(site_map, bins, weights=weights)
    else:
        numer = 0.0
        for kk in range(ni):
            for ll in range(nj):
                numer += _cross_genotype_D_plus(
                    genotypes_i[:, [kk], :], 
                    genotypes_j[:, [ll], :], 
                    site_map, 
                    bins
                )
        D_plus = numer / (ni * nj)
    return D_plus


def _cross_genotype_D_plus_between(
    left_genotypes_i,
    left_genotypes_j,
    right_genotypes_i,
    right_genotypes_j,
    left_map,
    right_map,
    bins
):
    """
    The one-population unphased ``D+`` estimator for use between two genomic 
    blocks. When there are >1 diploids in one of the populations, returns an 
    average over the ``n_i * n_j`` between-diploid pairs (where ``n`` is a 
    count of diploid samples)
    """
    ni = left_genotypes_i.shape[1]
    nj = left_genotypes_j.shape[1]
    if ni == 1 and nj == 1:
        left_weights = _compute_pi_xy(
            left_genotypes_i[:, 0], left_genotypes_j[:, 0])
        right_weights = _compute_pi_xy(
            right_genotypes_i[:, 0], right_genotypes_j[:, 0])
        D_plus = _count_locus_pairs_between(
            left_map, 
            right_map, 
            bins,
            left_weights=left_weights, 
            right_weights=right_weights
        )
    else:
        numer = 0.0
        for kk in range(ni):
            for ll in range(nj):
                numer += _cross_genotype_D_plus(
                    left_genotypes_i[:, [kk], :],
                    left_genotypes_j[:, [ll], :],
                    right_genotypes_i[:, [kk], :],
                    right_genotypes_j[:, [ll], :],
                    left_map,
                    right_map,
                    bins
                )
        D_plus = numer / (ni * nj)
    return D_plus


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
        print(utils._current_time(), 'Empty window: returning 0')
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
                print(utils._current_time(), 
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
                print(utils._current_time(), 
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
        print(utils._current_time(), 'Empty windows: returning 0')
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
                print(utils._current_time(), 
                    f"locus pairs summed (between) in bin {i}")
    else:
        edge0 = np.searchsorted(right_map, left_map + bins[0])
        for i, b in enumerate(bins[1:]):
            edge1 = np.searchsorted(right_map, left_map + b)
            sums[i] = (edge1 - edge0).sum() 
            edge0 = edge1
            if verbose:
                print(utils._current_time(), 
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
    map_coords = utils._map_function(r) * np.arange(L)
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
        coords, map_coords = utils._read_hapmap_map(
            filename, map_col=map_col, pos_col=pos_col)
    elif ".bed" or ".bedgraph" in filename:
        coords, map_coords = utils._read_bedgraph_map(
            filename, map_col=map_col, sep=map_sep)
    else:
        try:
            coords, map_coords = utils._read_hapmap_map(
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
    elif filename.endswith('.npy'):
        tot_map = np.load(filename)
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
        regions, _ = utils._read_bed_file(bed_file)
        mask = utils._regions_to_mask(regions)
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
        regions, _ = utils._read_bed_file(bed_file)
        mask = utils._regions_to_mask(regions)
    else: 
        mask = None

    vcf_str = ts.as_vcf(position_transform=utils._increment1, 
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
                print(utils._current_time(),
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

