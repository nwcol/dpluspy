"""
Houses functions for estimating ``D+`` from sequence data.
"""

from collections import defaultdict
import copy
import gzip
import numpy as np
import pandas
import re
import scipy
import warnings

from . import utils


def parse_statistics(
    vcf_file,
    bed_file,
    pop_file=None,
    pop_mapping=None,
    rec_map_file=None,
    pos_col="Position(bp)",
    map_col="Map(cM)",
    interp_method='linear',
    r=None,
    r_bins=None,
    phased=False,
    cross_pop=True,
    mut_map_file=None,
    mut_map_col=None,
    regions=None,
    regions_file=None,
    chrom=None
):
    """
    From a VCF and a BED file, compute ``D+`` and ``H`` statistics and their
    respective denominators given a configuration of genomic windows. Returns 
    a dictionary of region statistics. Each set of region statistics is stored
    in a dictionary with keys 'denoms', 'sums', 'pop_ids', 'bins' and optionally
    'mut_facs'.
    """
    if not (regions is not None) ^ (regions_file is not None):
        raise ValueError('You must provide either `regions` or `regions_file`')
    
    if regions_file:
        regions = np.loadtxt(regions_file)
    for region in regions:
        assert len(region) >= 2 and len(region) <= 3

    if chrom is None:
        chrom = ""

    if isinstance(r_bins, str):
        r_bins = np.loadtxt(r_bins)

    stats = {}

    for ii, region in enumerate(regions):
        region_stats = {}
        if len(region) == 2:
            interval = region
            interval_between = None 
        else:
            interval = region[:2]
            if region[1] != region[2]:
                interval_between = ((region[0], region[1]), (region[1], region[2]))
            else:
                interval_between = None

        # compute denominators
        denoms = compute_denominators(
            bed_file=bed_file,
            rec_map_file=rec_map_file,
            pos_col=pos_col,
            map_col=map_col,
            interp_method=interp_method,
            r=r,
            interval=interval,
            r_bins=r_bins
        )
        if interval_between:
            denoms += compute_denominators(
                bed_file=bed_file,
                rec_map_file=rec_map_file,
                pos_col=pos_col,
                map_col=map_col,
                interp_method=interp_method,
                r=r,
                interval_between=interval_between,
                r_bins=r_bins 
            )
        region_stats['denoms'] = denoms
        num_sites = denoms[-1]
        region_stats['num_sites'] = num_sites
        if num_sites == 0:
            print(utils._current_time(), 
                  f'Chromosome {chrom} window {ii} is empty')
            continue

        if mut_map_file:
            mut_facs, _ = compute_mutation_factors(
                mut_map_file,
                bed_file=bed_file,
                rec_map_file=rec_map_file,
                pos_col=pos_col,
                map_col=map_col,
                interp_method=interp_method,
                r=r,
                interval=interval,
                r_bins=r_bins,
                mut_map_col=mut_map_col
            )
            if interval_between:
                mut_facs += compute_mutation_factors(
                    mut_map_file,
                    bed_file=bed_file,
                    rec_map_file=rec_map_file,
                    pos_col=pos_col,
                    map_col=map_col,
                    interp_method=interp_method,
                    r=r,
                    interval_between=interval_between,
                    r_bins=r_bins,
                    mut_map_col=mut_map_col
                )[0]
            region_stats['mut_facs'] = mut_facs

        # Compute sums
        sums, pop_ids = compute_statistics(
            vcf_file,
            bed_file=bed_file,
            pop_file=pop_file,
            pop_mapping=pop_mapping,
            rec_map_file=rec_map_file,
            pos_col=pos_col,
            map_col=map_col,
            interp_method=interp_method,
            r=r,
            interval=interval,
            r_bins=r_bins,
            phased=phased,
            cross_pop=cross_pop
        )
        if interval_between:
            _sums = compute_statistics(
                vcf_file,
                bed_file=bed_file,
                pop_file=pop_file,
                pop_mapping=pop_mapping,
                rec_map_file=rec_map_file,
                pos_col=pos_col,
                map_col=map_col,
                interp_method=interp_method,
                r=r,
                interval_between=interval_between,
                r_bins=r_bins,
                phased=phased,
                cross_pop=cross_pop
            )[0]
            if np.sum(_sums) == 0:
                print(utils._current_time(),
                    f'Chromosome {chrom} window overhang {ii} has zero sum')
            sums += _sums
        if np.sum(sums) == 0:
            print(utils._current_time(),
                  f'Chromosome {chrom} window {ii} has zero sum')

        region_stats['pop_ids'] = pop_ids
        region_stats['sums'] = sums
        if isinstance(r_bins, str):
            r_bins = np.loadtxt(r_bins)
        region_stats['bins'] = r_bins
        stats[ii] = region_stats

        print(utils._current_time(), 
            f'Parsed chromosome {chrom} window {ii}')
        
    return stats


def compute_statistics(
    vcf_file,
    bed_file=None,
    pop_file=None,
    pop_mapping=None,
    rec_map_file=None,
    pos_col="Position(bp)",
    map_col="Map(cM)",
    interp_method='linear',
    r=None,
    interval=None,
    interval_between=None,
    r_bins=None,
    phased=False,
    cross_pop=True
):
    """
    Compute ``D+`` and ``H`` statistics from a VCF file. Returns a list of 
    population IDs and an array of statistic sums. 

    One may estimate ``D+`` for the set of site pairs within a single window
    by default or by using the argument `interval` to specify a region. One
    can also estimate ``D+`` for the set of site pairs spanning two windows,
    where the left locus is restricted to the lower-indexed window and the 
    right locus to the higher-indexed one, by specifying `interval_between`.

    :param vcf_file: Pathname of the VCF file to parse from.
    :param bed_file: Optional BED file defining the intervals to parse in 
        (default None).
    :param pop_file: Optional whitespace-separated file mapping sample IDs to
        populations (default None treats each sample as a unique population).
    :param pop_mapping: Optional dictionary mapping population IDs to lists of
        sample IDs (default None). Mutually exclusive with `pop_file`.
    :param rec_map_file: Optional recombination map in HapMap or BEDGRAPH
        format. Either `rec_map_file` or `r` and `L` must be provided.
    :param pos_col: Recombination map position column to use (default 
        "Position(bp)"). Functions only when a hapmap-format map is given.
    :param pos_col: Recombination map coordinate column to use (default 
        "Map(cM)"). 
    :param interp_method: Optional method for interpolating recombination map
        coordinates (default 'linear'). Must be a valid `kind` argument to
        `scipy.interpolate.interp1d`. 
    :param L: Optional chromosome length for constructing a uniform recomb. map
        when `r` is given (default None).
    :param r: Optional recombination rate per base pair for parsing with a 
        uniform recombination map (default None).
    :param interval: 2-tuple or list specifying the start end end of an 
        interval to parse (default None).
    :param interval_between: 2-tuple of 2-tuples, specifying the interval of 
        the left and right loci when parsing D+ between two windows 
        (default None). When given, the ``H`` row in the sum array is left as 0.
    :param r_bins: Recombination distance bin edges, given in units of ``r``. 
    :param phased: If True, treat all VCF data as phased and use the phased
        estimators for D+ (default False). 
    :param cross_pop: If True, compute cross-population statistics (default
        True).
    """
    if interval is not None or interval_between is not None:
        if interval is not None:
            within = True
            assert len(interval) == 2
            interval = (int(interval[0]), int(interval[1]))
            tot_interval = interval
        else:
            within = False
            assert len(interval_between) == 2
            left_interval = interval_between[0]
            right_interval = interval_between[1]
            assert len(left_interval) == len(right_interval) == 2
            assert right_interval[0] >= left_interval[1]
            interval_between = (
                (int(left_interval[0]), int(left_interval[1])),
                (int(right_interval[0]), int(right_interval[1])))
            tot_interval = (interval_between[0][0], interval_between[1][1])
        if tot_interval[0] < 1:
            raise ValueError('Interval start must be >= 1')
    else:
        within = True
        print(utils._current_time(), 'No interval given: Parsing all sites')

    if pop_file is not None and pop_mapping is not None:
        raise ValueError('You cannot use both `pop_file` and `pop_mapping`')
    if pop_file is not None:
        pop_mapping = _load_pop_file(pop_file)

    if r is not None:
        rec_map = _get_uniform_recombination_map(r, tot_interval[-1])
    else:
        rec_map = _load_recombination_map(
            rec_map_file, 
            pos_col=pos_col,
            map_col=map_col,
            interp_method=interp_method
        )

    if r_bins is None:
        raise ValueError('You must provide `r_bins`')
    if isinstance(r_bins, str):
        r_bins = np.loadtxt(r_bins)
    bins = utils._map_function(r_bins)

    if pop_mapping is not None:
        pop_ids = list(pop_mapping.keys())
    else:
        pop_ids = None

    if within:
        sites, genotype_matrix, sample_ids = _read_genotypes(
            vcf_file, bed_file=bed_file, interval=interval
        )
        if len(sites) == 0:
            warnings.warn('Loaded empty genotype array')
            zeros = _empty_sums(bins, pop_ids, cross_pop=cross_pop)
            return zeros, pop_ids
        pop_genotypes = _build_pop_genotypes(
            genotype_matrix, sample_ids, pop_mapping=pop_mapping
        )
        site_map = rec_map(sites)
        sums = _compute_stats_within(
            pop_genotypes, site_map, bins, cross_pop=cross_pop, phased=phased,
        )
    else:
        sites_left, genotype_matrix_left, sample_ids = _read_genotypes(
            vcf_file, bed_file=bed_file, interval=left_interval
        )
        sites_right, genotype_matrix_right, _ = _read_genotypes(
            vcf_file, bed_file=bed_file, interval=right_interval
        )
        if len(sites_left) == 0 or len(sites_right) == 0:
            warnings.warn('Loaded empty genotype array(s)')
            zeros = _empty_sums(bins, pop_ids, cross_pop=cross_pop)
            return zeros, pop_ids
        pop_genotypes_left = _build_pop_genotypes(
            genotype_matrix_left, sample_ids, pop_mapping=pop_mapping
        )
        pop_genotypes_right = _build_pop_genotypes(
            genotype_matrix_right, sample_ids, pop_mapping=pop_mapping
        )
        site_map_left = rec_map(sites_left)
        site_map_right = rec_map(sites_right)
        sums = _compute_stats_between(
            pop_genotypes_left, 
            pop_genotypes_right, 
            site_map_left, 
            site_map_right,
            bins, cross_pop=True, phased=False
        )
        pop_ids = list(pop_genotypes_left.keys())

    return sums, pop_ids


def _empty_sums(bins, pop_ids, cross_pop=True):
    """
    Create an array of zeros with shape determined by `bins` and `pop_ids`.
    """
    num_pops = len(pop_ids)
    if cross_pop:
        num_stats = (num_pops + num_pops ** 2) // 2
    else:
        num_stats = num_pops
    zeros = np.zeros((len(bins), num_stats), dtype=np.float64)

    return zeros


def compute_denominators(
    bed_file=None,
    rec_map_file=None,
    pos_col="Position(bp)",
    map_col="Map(cM)",
    interp_method='linear',
    r=None,
    interval=None,
    interval_between=None,
    r_bins=None
):
    """
    Compute the denominator for ``D+`` and ``H`` statistics. For ``H`` this is
    the number of callable sites- for ``D+`` it is the number of unique two-
    locus haplotypes.

    You must give either a `bed_file` or `L` to compute denominators. If an 
    interval is given and `L` is not, `L` is inferred from the end position of
    the interval.

    :param bed_file: Optional BED file (default None). Only sites within BED
        intervals are counted when given.
    :param rec_map_file: Optional recombination map file in HapMap or BEDGRAPH
        format (default None). 
    :param interp_method: Optional method for interpolating recombination map
        coordinates (default 'linear'). Must be a valid `kind` argument to
        `scipy.interpolate.interp1d`.
    :param r: Optional recombination rate for a uniform recombination map.
        You must provide `interval` or `interval_between` when using a uniform
        recombination map. 
    :param interval: 2-tuple or list specifying the start end end of a half-
        open contiguous interval (lower position inclusive, upper noninclusive,
        with both 1-indexed) within which to parse (default None).
    :param interval_between: 2-tuple of 2-tuples, specifying the interval of 
        the left and right loci when parsing denominators between two windows 
        (default None). When given, the ``H`` row in the array is left as 0.
    :param r_bins: Recombination distance bin edges, given in units of ``r``. 
    """
    if interval is not None or interval_between is not None:
        if interval is not None:
            within = True
            assert len(interval) == 2
            interval = (int(interval[0]), int(interval[1]))
            tot_interval = interval
        else:
            within = False
            assert len(interval_between) == 2
            left_interval = interval_between[0]
            right_interval = interval_between[1]
            assert len(left_interval) == len(right_interval) == 2
            assert right_interval[0] >= left_interval[1]
            interval_between = (
                (int(left_interval[0]), int(left_interval[1])),
                (int(right_interval[0]), int(right_interval[1])))
            tot_interval = (interval_between[0][0], interval_between[1][1])
        if tot_interval[0] < 1:
            raise ValueError('Interval start must be >= 1')
    else:
        if bed_file is None:
            raise ValueError('You must provide either an interval or BED file')
        within = True
        print(utils._current_time(), 'No interval given: Parsing all sites')
    
    if bed_file is not None:
        all_positions = utils._read_bed_file_positions(bed_file) + 1
    else:
        all_positions = np.arange(tot_interval[0], tot_interval[-1])
    
    if r_bins is None:
        raise ValueError('You must provide `r_bins`')
    if isinstance(r_bins, str):
        r_bins = np.loadtxt(r_bins)
    bins = utils._map_function(r_bins)

    if r is not None:
        rec_map = _get_uniform_recombination_map(r, all_positions[-1])
    else:
        rec_map = _load_recombination_map(
            rec_map_file, 
            pos_col=pos_col,
            map_col=map_col,
            interp_method=interp_method
        )

    if within:
        if interval is None:
            positions = all_positions
        else:
            positions = all_positions[
                (all_positions >= interval[0]) & (all_positions < interval[1])]
        pos_map = rec_map(positions)
        denoms = _count_locus_pairs(pos_map, bins)
        denoms = np.append(denoms, len(positions))
    else:
        left_positions = all_positions[
            (all_positions >= left_interval[0])
             & (all_positions < left_interval[1])]
        pos_map_left = rec_map(left_positions)
        right_positions = all_positions[
            (all_positions >= right_interval[0]) 
            & (all_positions < right_interval[1])]
        pos_map_right = rec_map(right_positions)
        denoms = _count_locus_pairs_between(pos_map_left, pos_map_right, bins)
        denoms = np.append(denoms, 0)

    return denoms
    

def compute_mutation_factors(
    mut_map_file,
    mut_map_col=None,
    bed_file=None,
    rec_map_file=None,
    pos_col="Position(bp)",
    map_col="Map(cM)",
    interp_method='linear',
    r=None,
    interval=None,
    interval_between=None,
    r_bins=None
):
    """
    Compute factors for weighting D+ by the local mutation rate given an 
    estimated mutation map. Returns an array of quantities ``uL * uR``, where 
    ``uL`` is the mutation rate at the left locus and ``uR`` that at the right,
    summed bin-wise. The last element of this array is the sum of ``uL``. Also
    returns the number of sites.
    
    :param mut_map_file: Mutation map file in BEDGRAPH format or stored as a 
        site-resolution .npy file.
    :param mut_map_col: Optional name for the column of the mutation map if a
        BEDGRAPH file is given (default None). When None, the rightmost column
        is used by default.
    :param bed_file: Optional BED file defining intervals to parse (default 
        None)
    :param rec_map_file: Optional recombination map file in HapMap or BEDGRAPH
        format (default None). A recombination map or `r` must be given.
    :param interp_method: Optional method for interpolating recombination map
        coordinates (default 'linear'). Must be a valid `kind` argument to
        `scipy.interpolate.interp1d`. 
    :param r: Optional recombination rate for a uniform recombination map. 
    :param interval: 2-tuple or list specifying the start end end of a 
        contiguous interval within which to parse (default None).
    :param interval_between: 2-tuple of 2-tuples, specifying the interval of 
        the left and right loci when parsing denominators between two windows 
        (default None). When given, the ``H`` row in the array is left as 0.
    :param r_bins: Recombination distance bin edges, given in units of ``r``. 
    """
    if interval is not None or interval_between is not None:
        if interval is not None:
            within = True
            assert len(interval) == 2
            interval = (int(interval[0]), int(interval[1]))
            tot_interval = interval
        else:
            within = False
            assert len(interval_between) == 2
            left_interval = interval_between[0]
            right_interval = interval_between[1]
            assert len(left_interval) == len(right_interval) == 2
            assert right_interval[0] >= left_interval[1]
            interval_between = (
                (int(left_interval[0]), int(left_interval[1])),
                (int(right_interval[0]), int(right_interval[1])))
            tot_interval = (interval_between[0][0], interval_between[1][1])
        if tot_interval[0] < 1:
            raise ValueError('Interval start must be >= 1')
    else:
        if bed_file is None:
            raise ValueError('You must provide either an interval or BED file')
        within = True
        print(utils._current_time(), 'No interval given: Parsing all sites')
    
    if bed_file is not None:
        all_positions = utils._read_bed_file_positions(bed_file) + 1
    else:
        all_positions = np.arange(tot_interval[0], tot_interval[-1])

    if r is not None:
        rec_map = _get_uniform_recombination_map(r, all_positions[-1])
    else:
        rec_map = _load_recombination_map(
            rec_map_file, 
            pos_col=pos_col,
            map_col=map_col,
            interp_method=interp_method
        )

    if r_bins is None:
        raise ValueError('You must provide `r_bins`')
    if isinstance(r_bins, str):
        r_bins = np.loadtxt(r_bins)
    bins = utils._map_function(r_bins)

    if within:
        if interval is None:
            positions = all_positions
        else:
            positions = all_positions[
                (all_positions >= interval[0]) & (all_positions < interval[1])
            ]
        pos_map = rec_map(positions)
        mut_map = _load_mutation_map(
            mut_map_file, positions, map_col=mut_map_col
        )
        mut_facs = _count_locus_pairs(pos_map, bins, weights=mut_map)
        sum_mut = np.sum(mut_map)
        mut_facs = np.append(mut_facs, sum_mut)
        num_sites = len(positions)
    else:
        positions_left = all_positions[
            (all_positions >= left_interval[0]) 
            & (all_positions < left_interval[1])
        ]
        pos_map_left = rec_map(positions_left)
        positions_right = all_positions[
            (all_positions >= right_interval[0]) 
            & (all_positions < right_interval[1])
        ]
        pos_map_right = rec_map(positions_right)      
        mut_map_left = _load_mutation_map(
            mut_map_file, positions_left, map_col=mut_map_col
        )
        mut_map_right = _load_mutation_map(
            mut_map_file, positions_right, map_col=mut_map_col
        )
        mut_facs = _count_locus_pairs_between(
            pos_map_left, 
            pos_map_right, 
            bins, 
            weights_l=mut_map_left, 
            weights_r=mut_map_right
        )
        mut_facs = np.append(mut_facs, 0)
        num_sites = 0

    return mut_facs, num_sites
    

def _load_pop_file(pop_file):
    """
    Load a population file.
    """
    pop_mapping = defaultdict(list)
    with open(pop_file, 'r') as fin:
        for line in fin:
            sample, pop = line.split()
            pop_mapping[pop].append(sample)

    return pop_mapping


def _build_pop_genotypes(genotypes, sample_ids, pop_mapping=None):
    """
    From an array of genotypes encoding 
    
    :param genotypes: Array of allelic states loaded by `_read_genotypes`. 
        This should have the shape ``(s, n, 2)``, where ``s`` is the number of
        sites and ``n`` is the number of diploid samples.
    :param sample_ids: VCF sample IDs loaded by `_read_genotypes`.
    :param pop_mapping: Dictionary, mapping population IDs to lists of sample
        IDs (default None). If None, each sample is placed in a unique
        population.
    
    returns: Dictionary that maps population IDs to population-specific
        genotype arrays.
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
                    sums[:-1, idx] = _haplotype_Dplus(Gt, site_map, bins)
                else:
                    sums[:-1, idx] = _genotype_Dplus(Gt, site_map, bins)
            else:
                if not cross_pop:
                    continue
                Gi = pop_genotypes[pop_i]
                Gj = pop_genotypes[pop_j]
                if phased:
                    sums[:-1, idx] = _cross_haplotype_Dplus(
                        Gi, Gj, site_map, bins
                    )
                else:
                    sums[:-1, idx] = _cross_genotype_Dplus(
                        Gi, Gj, site_map, bins
                    )
            idx += 1
    sums[-1] = _compute_pi(pop_genotypes, cross_pop=cross_pop)

    return sums


def _compute_stats_between(
    pop_genotypes_left,
    pop_genotypes_right,
    site_map_left,
    site_map_right,
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
    :param site_map_left: Site-wise array of recombination map coordinates for
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
                    sums[:-1, idx] = _haplotype_Dplus_between(
                        G_l, G_r, site_map_left, site_map_right, bins
                    )
                else:
                    sums[:-1, idx] = _genotype_Dplus_between(
                        G_l, G_r, site_map_left, site_map_right, bins
                    )
            else:
                if not cross_pop:
                    continue
                G_li = pop_genotypes_left[pop_i]
                G_lj = pop_genotypes_left[pop_j]
                G_ri = pop_genotypes_right[pop_i]
                G_rj = pop_genotypes_right[pop_j]
                if phased:
                    sums[:-1, idx] = _cross_haplotype_Dplus_between(
                        G_li, G_lj, G_ri, G_rj, 
                        site_map_left, site_map_right, bins
                    )
                else:
                    sums[:-1, idx] = _cross_genotype_Dplus_between(
                        G_li, G_lj, G_ri, G_rj, 
                        site_map_left, site_map_right, bins
                    )
            idx += 1
    sums[-1] = 0

    return sums


## Estimators of ``D+``


def _haplotype_Dplus(haplotypes, site_map, bins):
    """
    The one-population phased estimator for contiguous genomic regions. Calls
    itself recursively and returns a mean across haplotype pairs when there 
    are >2 haplotypes.
    """
    n = haplotypes.shape[1]
    if n == 2:
        weights = haplotypes[:, 0] != haplotypes[:, 1]
        Dplus = _count_locus_pairs(site_map, bins, weights=weights)
    else:
        numer = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                numer += _haplotype_Dplus(haplotypes[:, [i, j]], site_map, bins)
        Dplus = numer / (n * (n - 1) / 2)

    return Dplus


def _haplotype_Dplus_between(
    haplotypes_l, 
    haplotypes_r, 
    site_map_l, 
    site_map_r, 
    bins
):
    """
    The one-population phased estimator between two genomic regions. When there
    are >2 haplotypes in the sample, calls itself recursively and returns a 
    mean across n choose 2 haplotype pairs.
    """
    n = haplotypes_l.shape[1]
    if n == 2:
        weights_l = haplotypes_l[:, 0] != haplotypes_l[:, 1]
        weights_r = haplotypes_r[:, 0] != haplotypes_r[:, 1]
        Dplus = _count_locus_pairs_between(
            site_map_l,
            site_map_r, 
            bins, 
            weights_l=weights_l,
            weights_r=weights_r
        )
    else:
        numer = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                numer += _haplotype_Dplus_between(
                    haplotypes_l[:, [i]], 
                    haplotypes_r[:, [j]], 
                    site_map_l,
                    site_map_r, 
                    bins
                )
        Dplus = numer / (n * (n - 1) / 2)

    return Dplus


def _cross_haplotype_Dplus(haplotypes_i, haplotypes_j, site_map, bins):
    """
    The cross-population phased estimator within a genomic region. When one or
    more samples has >1 haplotypes, calls itself recursively and returns an 
    average across the ``n_i * n_j`` unique haplotype pairs.
    """
    ni = haplotypes_i.shape[1]
    nj = haplotypes_j.shape[1]
    if ni == 1 and nj == 1:
        weights = haplotypes_i[:, 0] != haplotypes_j[:, 0]
        Dplus = _count_locus_pairs(site_map, bins, weights=weights)
    else:
        numer = 0.0
        for k in range(ni):
            for l in range(nj):
                numer += _cross_haplotype_Dplus(
                    haplotypes_i[:, [k]], haplotypes_j[:, [l]], site_map, bins
                )
        Dplus = numer / (ni * nj)

    return Dplus
 

def _cross_haplotype_Dplus_between(
    haplotypes_li, 
    haplotypes_lj,
    haplotypes_ri, 
    haplotypes_rj,
    site_map_l, 
    site_map_r, 
    bins
):
    """
    The cross-population phased between-genomic-blocks ``D+`` estimator. 
    """
    ni = haplotypes_li.shape[1]
    nj = haplotypes_lj.shape[1]
    if ni == 1 and nj == 1:
        weights_l = haplotypes_li[:, 0] != haplotypes_lj[:, 1]
        weights_r = haplotypes_ri[:, 0] != haplotypes_rj[:, 1]
        Dplus = _count_locus_pairs_between(
            site_map_l,
            site_map_r, 
            bins, 
            weights_l=weights_l,
            weights_r=weights_r
        )
    else:
        numer = 0.0
        for k in range(ni):
            for l in range(nj):
                numer += _haplotype_Dplus_between(
                    haplotypes_li[:, [k]], 
                    haplotypes_lj[:, [l]],
                    haplotypes_ri[:, [k]], 
                    haplotypes_rj[:, [l]],
                    site_map_l,
                    site_map_r, 
                    bins
                )
        Dplus = numer / (ni * nj)

    return Dplus


def _genotype_Dplus(genotypes, site_map, bins):
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
        Dplus = _count_locus_pairs(site_map, bins, weights=weights)
    else:
        numer = 0.0
        for i in range(n):
            numer += _genotype_Dplus(genotypes[:, [i], :], site_map, bins)
        Dplus = numer / n

    return Dplus


def _genotype_Dplus_between(
    genotypes_l,
    genotypes_r,
    site_map_l,
    site_map_r,
    bins
):
    """
    The one-population, unphased, between-genomic-blocks ``D+`` estimator. 
    When there are >1 diploids in the sample, returns an average over ``n`` 
    within-diploid estimates (where ``n`` is a count of diploid samples).
    """
    n = genotypes_l.shape[1]
    if n == 1:
        weights_l = genotypes_l[:, 0, 0] != genotypes_l[:, 0, 1]
        weights_r = genotypes_r[:, 0, 0] != genotypes_r[:, 0, 1]
        Dplus = _count_locus_pairs_between(
            site_map_l, 
            site_map_r, 
            bins,
            weights_l=weights_l, 
            weights_r=weights_r
        )
    else:
        numer = 0.0
        for i in range(n):
            numer += _genotype_Dplus_between(
                genotypes_l[:, [i], :], 
                genotypes_r[:, [i], :],
                site_map_l,
                site_map_r,
                bins
            )
        Dplus = numer / n

    return Dplus


def _cross_genotype_Dplus(genotypes_i, genotypes_j, site_map, bins):
    """
    The one-population unphased ``D+`` within-block estimator. When there are 
    >1 diploids in one of the populations, returns an average over the 
    ``n_i * n_j`` between-diploid pairs (where ``n`` is a count of diploid
    samples)
    """
    ni = genotypes_i.shape[1]
    nj = genotypes_j.shape[1]
    if ni == 1 and nj == 1:
        weights = _pi_xy(genotypes_i[:, 0], genotypes_j[:, 0])
        Dplus = _count_locus_pairs(site_map, bins, weights=weights)
    else:
        numer = 0.0
        for k in range(ni):
            for l in range(nj):
                numer += _cross_genotype_Dplus(
                    genotypes_i[:, [k], :], 
                    genotypes_j[:, [l], :], 
                    site_map, 
                    bins
                )
        Dplus = numer / (ni * nj)

    return Dplus


def _cross_genotype_Dplus_between(
    genotypes_li,
    genotypes_lj,
    genotypes_ri,
    genotypes_rj,
    site_map_l,
    site_map_r,
    bins
):
    """
    The one-population unphased ``D+`` estimator for use between two genomic 
    blocks. When there are >1 diploids in one of the populations, returns an 
    average over the ``n_i * n_j`` between-diploid pairs (where ``n`` is a 
    count of diploid samples)
    """
    ni = genotypes_li.shape[1]
    nj = genotypes_lj.shape[1]
    if ni == 1 and nj == 1:
        weights_l = _pi_xy(genotypes_li[:, 0], genotypes_lj[:, 0])
        weights_r = _pi_xy(genotypes_ri[:, 0], genotypes_rj[:, 0])
        Dplus = _count_locus_pairs_between(
            site_map_l, 
            site_map_r, 
            bins,
            weights_l=weights_l, 
            weights_r=weights_r
        )
    else:
        numer = 0.0
        for k in range(ni):
            for l in range(nj):
                numer += _cross_genotype_Dplus(
                    genotypes_li[:, [k], :],
                    genotypes_lj[:, [l], :],
                    genotypes_ri[:, [k], :],
                    genotypes_rj[:, [l], :],
                    site_map_l,
                    site_map_r,
                    bins
                )
        Dplus = numer / (ni * nj)

    return Dplus


def _pi_xy(genotypes_i, genotypes_j):
    """
    Compute the pairwise divergence between two diploids. This is the nucleotide 
    diversity, conditional on sampling one allele copy from each diploid.

    :param genotypes_i: Array of allelic states with shape (s, 2) for diploid i.
    :param genotypes_j: Array of allelic states for diploid j with shape (s, 2)

    :returns: Array of pi_ij with shape (s,)
    :rtype: np.ndarray 
    """
    pairwise_diff = genotypes_i[:, :, np.newaxis] != genotypes_j[:, np.newaxis]
    pi = pairwise_diff.sum((2, 1)) / 4

    return pi


## Locus pair-counting functions


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
    site_map_l, 
    site_map_r, 
    bins, 
    weights_l=None, 
    weights_r=None,
    verbose=False
):
    """
    Compute binned counts of locus pairs between two discontinuous genomic
    windows.

    Used to compute D+ and its denominator. 

    :params sitemap1, sitemap2: Array giving the recombination map coordinates 
        of sites in linear units (cM or M) for the left and right windows.
    :param bins: Array of recombination bin edges, given in the same unit as 
        the map (cM or M). 
    :params weights1, weights1: Array with lengths equal to `sitemap1` and 
        `sitemap2` respectively, assigning weight site in each block 
        (default None).
    :returns: Array of binned locus pair counts.
    """
    num_bins = len(bins) - 1
    sums = np.zeros(num_bins, dtype=np.float64)

    if len(site_map_l) == 0 or len(site_map_r) == 0:
        print(utils._current_time(), 'Empty windows: returning 0')
        return sums
    if not np.all(np.diff(site_map_l) >= 0):
        raise ValueError('`site_map_l` must increase monotonically')
    if not np.all(np.diff(site_map_r) >= 0):
        raise ValueError('`site_map_r` must increase monotonically')
    
    if site_map_l[-1] > site_map_r[0]:
        raise ValueError(
            '`site_map_r` must have higher coords than `site_map_l`')
    if (weights_l is not None) ^ (weights_r is not None):
        raise ValueError('You must provide weights for both windows')
    if weights_l is not None:
        if len(weights_l) != len(site_map_l):
            raise ValueError("Map and weight lengths mismatch for block 1")
        if len(weights_r) != len(site_map_r):
            raise ValueError("Map and weight lengths mismatch for block 2")

    num_bins = len(bins) - 1

    if weights_l is not None:
        indices = np.searchsorted(site_map_r, site_map_l + bins[0])
        assert np.all(indices >= 0)
        cum_weights2 = np.concatenate(([0], np.cumsum(weights_r)))
        cum_sum0 = cum_weights2[indices]
        for i, b in enumerate(bins[1:]):
            indices = np.searchsorted(site_map_r, site_map_l + b)
            assert np.all(indices >= 0)
            cum_sum1 = cum_weights2[indices]
            sums[i] = (weights_l * (cum_sum1 - cum_sum0)).sum()
            cum_sum0 = cum_sum1
            if verbose:
                print(utils._current_time(), 
                    f"locus pairs summed (between) in bin {i}")
    else:
        edge0 = np.searchsorted(site_map_r, site_map_l + bins[0])
        for i, b in enumerate(bins[1:]):
            edge1 = np.searchsorted(site_map_r, site_map_l + b)
            sums[i] = (edge1 - edge0).sum() 
            edge0 = edge1
            if verbose:
                print(utils._current_time(), 
                    f"locus pairs summed (between) in bin {i}")

    return sums


## Utilities


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
    mapfunc = scipy.interpolate.interp1d(
        coords, 
        map_coords, 
        kind='nearest', 
        bounds_error=False, 
        fill_value=(map_coords[0], map_coords[-1])
    )
    return mapfunc


def _load_recombination_map(
    filename, 
    pos_col="Position(bp)",
    map_col="Map(cM)",
    interp_method="linear", 
    unit='cM',
    sep=None,
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
        coords, map_coords = utils._read_hapmap_map(
            filename, map_col=map_col, pos_col=pos_col
        )
    elif ".bed" or ".bedgraph" in filename:
        coords, map_coords = utils._read_bedgraph_map(
            filename, map_col=map_col, sep=sep
        )
    else:
        try:
            coords, map_coords = utils._read_hapmap_map(
                filename, map_col=map_col, pos_col=pos_col
            )
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
    map_func = scipy.interpolate.interp1d(
        xs, 
        ys, 
        kind=interp_method,
        bounds_error=False,
        fill_value=(ys[0], ys[-1])
    )
    return map_func


def _load_mutation_map(filename, positions, map_col="mut_map"):
    """
    Load a mutation map in BEDGRAPH format, or from a site-resolution .npy
    file.

    :param filename: Pathname of the mutation map file.
    :param positions: Array of 1-indexed positions for which to load rates.

    :returns: Site-resolution mutation map array.
    """
    if ".bedgraph" in filename or ".csv" in filename:
        data = pandas.read_csv(filename)
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


def _read_genotypes(
    vcf_file, 
    bed_file=None, 
    multiallelic=False,
    missing_to_ref=False,
    filtered=False,
    interval=None,
    verbose=0
):
    """
    Read sites and genotypes from a VCF file.

    If return_dict is True, returns a dictionary mapping sites to site genotype
    arrays with shapes (n, 2) where n is the number of samples. Otherwise,
    returns an array of sites, an array of genotypes with shape (s, n, 2) where
    s is the number of sites. This object will occupy a large amount of memory
    when the sample size is large- intended usage is for sample sizes on the 
    order of one or two dozen diploids.

    We encode genotypes represented as `A1/A2` or `A1|A2` in the file in the 
    form `[A1, A2]`. Thus if data is phased, the genotype array can be 
    converted into a haplotype array by flattening over the last axis. Also, 
    multiallelic sites are easily represented in this format, e.g. as
    `[[0, 1], [0, 2]]`.

    :param vcf_file: Pathname of a VCF file
    :param bed_file: Filename for BED mask to impose on sites (default None).
    :param multiallelic: If True, do not skip multiallelic sites 
        (default False).
    :param missing_to_ref: If True, genotypes ./. and .|. will be read as 0/0
        or 0|0 respectively (default False skips sites with any missing data).
    :param interval: 2-tuple or list specifying 1-indexed upper and lower 
        bounds on positions, where the upper bound is noninclusive 
        (default None).
    :param verbose: If > 0, print a progress message every `verbose` lines.

    :returns: Array of 1-indexed sites, array of genotypes, list of sample IDs
    """
    if bed_file is not None:
        regions, _ = utils._read_bed_file(bed_file)
        mask = utils._regions_to_mask(regions)
        masked = True
    else:
        masked = False

    if interval is not None:
        intervaled = True
    else:
        intervaled = False

    if vcf_file.endswith('.gz'):
        open_func = gzip.open 
    else:
        open_func = open

    _sites = []
    _genotypes = []
    counter = 0

    with open_func(vcf_file, 'rb') as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    sample_ids = line.split()[9:]
                continue
            split_line = line.split()
            pos1 = int(split_line[1])
            if verbose > 1:
                if counter % verbose == 0 and counter > 1:
                    print(utils._current_time(),
                        f'parsed position {pos1} line {counter}')
            counter += 1
            if intervaled:
                if pos1 < interval[0]:
                    continue
                if pos1 >= interval[1]:
                    break
            if masked:
                pos0 = pos1 - 1
                if pos0 >= len(mask):
                    break
                if mask[pos0] == True:
                    continue

            if filtered:
                filtr = split_line[6]
                if filtr not in ('PASS', '.'):
                    continue
            
            ref = split_line[3]
            alts = split_line[4].split(',')
            alleles = [ref] + alts

            if np.any([len(allele) > 1 for allele in alleles]):
                continue
            if not multiallelic:
                if len(alts) > 1:
                    continue

            split_samples = [sample.split(':') for sample in split_line[9:]]
            genotype_strs = [sample[0] for sample in split_samples]
            genotype_list = [re.split("/|\\|", gt) for gt in genotype_strs]
            skip_line = False
            for i, gt in enumerate(genotype_list):
                for j, allele in enumerate(gt):
                    if allele == '.':
                        if missing_to_ref:
                            genotype_list[i][j] = '0'
                        else:
                            skip_line = True
            if skip_line:
                continue
            _genotypes.append(np.array(genotype_list))
            _sites.append(pos1)

    sites = np.array(_sites, dtype=np.int64)
    genotypes = np.array(_genotypes, dtype=np.int64)

    return sites, genotypes, sample_ids
