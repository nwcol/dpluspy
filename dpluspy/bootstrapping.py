"""
Houses functions for performing bootstraps and computing means.
"""

from collections import defaultdict
import copy
import numpy as np
import pickle
import warnings

from . import utils


## Subsetting empirical statistics


def subset_statistics(
    statistics, 
    to_pops=None, 
    min_r=None, 
    max_r=None,
    return_dict=False
):
    """
    Subset a dictionary holding statistics by populations or bins. 

    :param statistics: A dictionary with fields 'means', 'varcovs', 'pop_ids',
        and 'bins'.
    :type statistics: dict
    :param to_pops: List of population IDs to subset to (default None).
    :param min_r: Minimum lower bin edge, inclusive (default None).
    :param max_r: Maximum upper bin edge, inclusive (default None).
    :param return_dict: If True, return a dictionary with the same fields as
        required for the input- otherwise return bins, means and varcovs in a 
        tuple (default False).

    :returns: Dictionary of subsetted statistics.
    :rtype: dict
    """
    means = statistics['means']
    varcovs = statistics['varcovs']
    pop_ids = statistics['pop_ids']
    bins = statistics['bins']
    if to_pops is None:
        to_pops = pop_ids
    for pop_id in to_pops:
        if pop_id not in pop_ids:
            raise ValueError(f'"{pop_id}" is not represented in the data')
    if min_r is not None or max_r is not None:
        if min_r is not None:
            min_idx = np.where(bins >= min_r)[0][0]
        else:
            min_idx = 0
        if max_r is not None:
            max_idx = np.where(bins <= max_r)[0][-1]
        else:
            max_idx = len(bins) - 1
        means = means[min_idx:max_idx] + [means[-1]]
        varcovs = varcovs[min_idx:max_idx] + [varcovs[-1]]
        bins = bins[min_idx:max_idx + 1]
    else:
        bins = statistics['bins']
    new_means = subset_means(means, pop_ids, to_pops)
    new_varcovs = subset_varcovs(varcovs, pop_ids, to_pops)

    if return_dict:
        if to_pops:
            pop_ids = to_pops
        subset_stats = {
            'pop_ids': pop_ids,
            'bins': bins,
            'means': new_means,
            'varcovs': new_varcovs
        }
        return subset_stats
    else:
        return bins, new_means, new_varcovs


def load_statistics(filename, to_pops=None):
    # deprecated
    stats = pickle.load(open(filename, "rb"))
    bins = stats["bins"]
    if to_pops is not None:
        all_ids = stats["pop_ids"]
        means = subset_means(stats["means"], all_ids, to_pops)
        varcovs = subset_varcovs(stats["varcovs"], all_ids, to_pops)
        pop_ids = to_pops
    else:
        pop_ids = stats["pop_ids"]
        means = stats["means"]
        varcovs = stats["varcovs"]

    return pop_ids, bins, means, varcovs


def subset_means(means, pop_ids, to_pops):
    """
    Subset a list of binned statistics representing `pop_ids` to `to_pops`. The
    returned statistics will be in an order determined by `to_pops`.

    :param means: List of 1d arrays to subset.  
    :type means: list of np.ndarray
    :param pop_ids: List of populations represented in `means`.
    :type pop_ids: list of str
    :param to_pops: List of populations to subset to. One and two-population
        statistics from this list will be returned.
    :typr to_pops: list of str

    :returns: A list of 1d arrays subset to `to_pops`.
    :rtype: list of np.ndarray
    """
    for pop in to_pops:
        if pop not in pop_ids:
            raise ValueError(f'"{pop}" not in `pop_ids`')
    stats = utils._get_Dplus_names(list(range(len(pop_ids))))
    to_pop_idx = [pop_ids.index(pop) for pop in to_pops]
    to_stats = []
    for i, idx0 in enumerate(to_pop_idx):
        for idx1 in to_pop_idx[i:]:
            _idx0, _idx1 = sorted([idx0, idx1])
            to_stats.append(f"D+_{_idx0}_{_idx1}")
    to_idx = np.array([stats.index(to_stat) for to_stat in to_stats])
    new_means = [means[i][to_idx] for i in range(len(means))]

    return new_means


def subset_varcovs(varcovs, pop_ids, to_pops):
    """
    Marginalize a list of bin-wise covariance matrices from `pop_ids` to `pops`.

    :returns: A list of 2d covariance matrices subset to `to_pops`.
    :rtype: list of np.ndarray
    """
    for pop in to_pops:
        if pop not in pop_ids:
            raise ValueError(f'"{pop}" not in `pop_ids`')
    stats = utils._get_Dplus_names(list(range(len(pop_ids))))
    to_pop_idx = [pop_ids.index(pop) for pop in to_pops]
    to_stats = []
    for i, idx0 in enumerate(to_pop_idx):
        for idx1 in to_pop_idx[i:]:
            _idx0, _idx1 = sorted([idx0, idx1])
            to_stats.append(f"D+_{_idx0}_{_idx1}")
    to_idx = np.array([stats.index(to_stat) for to_stat in to_stats])
    mesh = np.ix_(to_idx, to_idx)
    new_varcovs = [varcovs[i][mesh] for i in range(len(varcovs))]

    return new_varcovs


## Loading statistics


def load_raw_stats(filenames, load_mut_facs=False):
    """
    Load statistics from .pkl files. The expected format of each file is
    {
        "key1": {
            "sums": array([[...]]),
            "denoms": array([...])},
            "bins": array([...]),
            "pop_ids": [...], 
            "mut_facs": array([]) (optional)
            ...
        },
        "key2": {...},
        ...
    }

    :param filenames: Files from which to load raw statistics.

    :returns: population IDs, bin edges, and dictionary of raw data
    :rtypes: (list, np.ndarray, dict)
    """
    raw_data = defaultdict(dict)
    for filename in filenames:
        contents = pickle.load(open(filename, "rb+"))
        for key in contents:
            if key in raw_data: 
                raise ValueError(f"{key} appears twice in input")
            bins = contents[key]["bins"]
            pop_ids = contents[key]["pops"]
            raw_data[key]["sums"] = contents[key]["sums"]
            raw_data[key]["denoms"] = contents[key]["denoms"]
            raw_data[key]["weights"] = contents[key]["weights"]
            if load_mut_facs:
                if not "mut_facs" in contents[key]:
                    raise ValueError(f"Region {key} has no `mut_facs`")
                raw_data[key["mut_facs"]] = contents[key]["mut_facs"]

    return pop_ids, bins, raw_data


def load_bootstrap_means(filename, to_pops=None, size=None):
    # load list of bootstrap means
    with open(filename, "rb") as fin:
        contents = pickle.load(fin)
    if size is not None:
        all_means = contents["bootstrap_means"]
        samples = np.random.choice(
            np.arange(len(all_means)), size=size, replace=False
        )
        bootstrap_means = [all_means[i] for i in samples]
    else:
        bootstrap_means = contents["bootstrap_means"]
    pop_ids = contents["pop_ids"]
    if to_pops is None:
        ret_means = bootstrap_means
        ret_pop_ids = pop_ids
    else:
        ret_means = []
        for means in bootstrap_means:
            ret_means.append(utils.subset_means(means, pop_ids, to_pops))
        ret_pop_ids = to_pops
    bins = contents["bins"]

    return ret_means, bins, ret_pop_ids


def bootstrap(regions, num_reps=1000):
    """
    Perform a bootstrap to obtain covariance matrices for D+ and H statistics,
    estimated in genomic blocks. Operates upon sums of D+, H, and their
    respective denominators, so that regions are weighted appropriately. 
    
    :param regions: A dictionary with sums of D+ and H statistics from genomic
        regions as values, with arbitrary keys specifying region names.
    :type regions: dict
    :param num_reps: Number of bootstrap replicates to perform; if None, then 
        uses the number of regions (default None).
    :type num_reps: int
    
    :returns: Lists of mean and covariance arrays for each D+ bin and for the H
        statistics. Optionally also a list of bootstrap replicate means.
    :rtype: tuple (2 or 3-tuple of lists)
    """
    means = means_across_regions(regions)
    labels = list(regions.keys())
    sample_size = len(regions)
    bootstrap_means = []
    for i in range(num_reps):
        samples = np.random.choice(labels, sample_size, replace=True)
        sampled_regions = [regions[sample] for sample in samples]
        _means = means_across_replicates(sampled_regions)
        bootstrap_means.append(_means)
    varcovs = []
    for i in range(len(means)):
        bin_means = np.array([_means[i] for _means in bootstrap_means])
        varcov_matrix = np.cov(bin_means.T)
        # this occurs when the bootstrap involves only one statistic
        if varcov_matrix.shape == ():
            varcov_matrix = varcov_matrix.reshape((1, 1))
        varcovs.append(varcov_matrix)

    return means, varcovs


def get_bootstrap_replicates(regions, num_reps=1000):
    """
    
    """

    bootstrap_means = []

    return bootstrap_means


def means_across_regions(regions):
    """
    Compute mean statistics across genomic windows.
    
    :param regions: Dictionary of dictionaries that represent genomic windows,
        each containing summed D+, H statistics and respective denominators.
        Statistics should be numpy arrays, with the 0th dimension indexing bins.
    :type regions: dict

    :returns: A list holding the mean statistics in each bin.
    :rtype: list
    """
    sums = 0.0
    denoms = 0.0
    for key in regions:
        sums += regions[key]["sums"]
        denoms += regions[key]["denoms"]
    ext_denoms = np.repeat(denoms[:, np.newaxis], sums.shape[1], axis=1)
    raw_means = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, ext_denoms, where=ext_denoms > 0, out=raw_means)
    if np.any(np.isnan(raw_means)):
        warnings.warn("nan means exist in output")
    means = [raw_means[i] for i in range(len(raw_means))]

    return means


def means_across_replicates(replicates):
    """
    Compute mean statistics across a list of replicates. 
    
    :param replicates: List of dictionaries that hold summed D+, H statistics
        and the respective denominators as numpy arrays.
    :type replicates: list

    :returns: A list of mean statistics for each bin.
    :rtype: list
    """
    rep_dict = {i: replicate for i, replicate in enumerate(replicates)}
    means = means_across_regions(rep_dict)

    return means


def weighted_bootstrap(
    regions,
    num_reps=None, 
    sample_size=None, 
    get_reps=False
):
    """
    Perform a bootstrap on a dictionary of region-specific D+ and H sums, 
    using the mutation rate-weighted estimator to de-distort the shape of the
    D+ curve. 

    Each region should be represented by a weighted dictionary in `regions`,
    minimally containing 'sums', 'num_sites' and 'mut_facs'. 
    """
    if num_reps is None:
        num_reps = len(regions)
    if sample_size is None:
        sample_size = len(regions)
    means = weighted_means_across_regions(regions)
    labels = list(regions.keys())
    sample_size = len(regions)
    bootstrap_means = []
    for i in range(num_reps):
        samples = np.random.choice(labels, sample_size, replace=True)
        sampled_regions = [regions[sample] for sample in samples]
        _means = weighted_means_across_replicates(sampled_regions)
        bootstrap_means.append(_means)
    varcovs = []
    for i in range(len(means)):
        bin_means = np.array([_means[i] for _means in bootstrap_means])
        varcov_matrix = np.cov(bin_means.T)
        # this occurs when the bootstrap involves only one statistic
        if varcov_matrix.shape == ():
            varcov_matrix = varcov_matrix.reshape((1, 1))
        varcovs.append(varcov_matrix)
    if get_reps:
        ret = (means, varcovs, bootstrap_means)
    else:
        ret = (means, varcovs)

    return ret


def weighted_means_across_regions(regions):
    """
    Compute mutation-rate weighted D+ across a dictionary of regions.
    """
    sums = 0.0
    pair_counts = 0.0
    mut_prods = 0.0
    num_sites = 0.0
    for key in regions:
        sums += regions[key]['sums']
        pair_counts += regions[key]['denoms'][:-1]
        mut_prods += regions[key]['mut_facs'][:-1]
        num_sites += regions[key]['denoms'][-1]
    # Compute the u-weighted denominator
    weighted_denoms = mut_prods / (mut_prods.sum() / pair_counts.sum())
    denoms = np.append(weighted_denoms, num_sites)
    
    ext_denoms = np.repeat(denoms[:, np.newaxis], sums.shape[1], axis=1)
    raw_means = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, ext_denoms, where=ext_denoms > 0, out=raw_means)
    if np.any(np.isnan(raw_means)):
        warnings.warn("nan means exist in output")
    means = [raw_means[i] for i in range(len(raw_means))]

    return means


def weighted_means_across_replicates(replicates):
    """
    Operates on a list of dictionaries; wraps `weighted_means_across_regions`.
    """
    rep_dict = {i: replicate for i, replicate in enumerate(replicates)}
    means = weighted_means_across_regions(rep_dict)

    return means


### DEPRECATED. this stuff doesn't work well!!


def compute_weighted_denoms(mut_facs, denoms):
    """
    Compute weighted denominators for D+.

    Weighting takes the form of adjusting the effective number of locus pairs
    (the denominator) of each bin in inverse proportion to its mean product
    of locus mutation rates u_L * u_R (relative to the average across bins).

    :param mut_facs: Binned sums of locus-pair mutation rates u_L * u_R. The 
        last element should be the sum of locus mutation rates, which is not 
        used here.
    :param denoms: Binned counts of locus pairs. It is assumed that the last
        element contains the denominator for H (the number of loci), which is
        ignored here and returned as the last element of `weights`.

    :returns: Weights for adjusting D+
    :rtype: np.ndarray
    """
    num_sites = denoms[-1]
    mut_prods = mut_facs[:-1]
    locus_pairs = denoms[:-1]
    #factor = mut_prods.sum() / locus_pairs.sum()
    factor = (mut_facs[-1] / num_sites) ** 2
    _weighted_denoms = mut_prods / factor 
    weighted_denoms = np.append(_weighted_denoms, num_sites)

    return weighted_denoms


def compute_weights(mut_facs, denoms):
    """
    Compute the ratio of the weighted D+ denominator over the unweighted one.
    """
    weighted_denoms = compute_weighted_denoms(mut_facs, denoms)
    weights = weighted_denoms / denoms

    return weights



