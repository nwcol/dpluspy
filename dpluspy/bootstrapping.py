"""
Functions for subsetting statistics, computing means, and bootstrapping.
"""

import random
import numpy as np
import pickle

from . import utils


def subset_stats(
    statistics, 
    to_pops=None, 
    min_r=None, 
    max_r=None,
    return_dict=True
):
    """
    Subset a dictionary holding statistics by population and/or r. 

    :param statistics: A dictionary with fields 'means', 'varcovs', 'pop_ids',
        and 'bins'.
    :type statistics: dict
    :param to_pops: List of population IDs to subset to (default None).
    :param min_r: Minimum lower bin edge, inclusive (default None).
    :param max_r: Maximum upper bin edge, inclusive (default None).
    :param return_dict: If True, return a dictionary with the same fields as
        required for the input- otherwise return bins, means and varcovs in a 
        tuple (default True).

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
    stats = utils._DP_names(list(range(len(pop_ids))))
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
    stats = utils._DP_names(list(range(len(pop_ids))))
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


def load_raw_stats(filenames):
    """
    Load statistics from .pkl files. 

    :param filenames: Files from which to load raw statistics.

    :returns: population IDs, bin edges, and dictionary of raw data
    :rtypes: (list, np.ndarray, dict)
    """
    regions = dict()
    for filename in filenames:
        with open(filename, "rb+") as fin:
            contents = pickle.load(fin)
        for label in contents:
            if label in regions: 
                raise ValueError(f"Label {label} appears twice in input")
            regions[label] = contents[label]
    return regions


def bootstrap_stats(
    regions, 
    num_reps=None, 
    weighted=False,
    aggregate=False
):
    """
    Perform a bootstrap to obtain covariance matrices for D+ and H statistics,
    estimated in genomic blocks. Operates upon sums of D+, H, and their
    respective denominators, so that regions are weighted appropriately. 
    
    :param dict regions: Dictionary of sums corresponding to genomic regions.
    :param int num_reps: Optional number of bootstrap replicates to assemble
        (defalts to `len(regions)`).
    :param bool weighted: If True (default False), compute mutation-rate 
        weighted statistics. Assumes "mut_facs" exists in each region.
    
    :returns: List of means and list of bootstrap covariances.
    """
    if num_reps is None:
        num_reps = len(regions)
    if weighted:
        means = weighted_means_across_regions(regions)
    else:
        means = means_across_regions(regions)
    bootstrap_means = get_bootstrap_reps(
        regions, num_reps=num_reps, weighted=weighted, aggregate=aggregate
    )
    varcovs = compute_varcovs(bootstrap_means)
    return means, varcovs


def get_bootstrap_reps(data, num_reps=None, weighted=False, aggregate=False):
    """
    Perform a bootstrap and return a list of replicate means.

    :param dict data: Dictionary of sums corresponding to genomic regions.
    :param int num_reps: Optional number of bootstrap replicates to assemble
        (defalts to `len(regions)`).
    :param bool weighted: If True (default False), compute mutation-rate 
        weighted statistics. Assumes "mut_facs" exists in each region.

    :returns list: Bootstrap replicate means
    """
    if num_reps is None:
        num_reps = len(data)
    labels = list(data.keys())
    sample_size = len(data)
    replicates = []
    for _ in range(num_reps):
        samples = random.choices(labels, k=sample_size)
        sampled_data = [data[sample] for sample in samples]
        if weighted:
            replicate = _weighted_means_across_replicates(
                sampled_data, aggregate=aggregate)
        else:
            replicate = _means_across_replicates(sampled_data)
        replicates.append(replicate)
    return replicates
 

def compute_varcovs(bootstrap_means):
    """
    Compute a list of variance-covariance matrices (one for each recombination 
    bin, and one for H) from a list of bootstrap replicate means.

    :param list bootstrap_means: A list of bootstrap means.
    """
    varcovs = []
    for i in range(len(bootstrap_means[0])):
        bin_means = np.array([_means[i] for _means in bootstrap_means])
        varcov_matrix = np.cov(bin_means.T)
        # This occurs when the bootstrap involves only one statistic
        if varcov_matrix.shape == ():
            varcov_matrix = varcov_matrix.reshape((1, 1))
        varcovs.append(varcov_matrix)
    return varcovs

def means_across_regions(regions):
    """
    Compute mean D+ and H across genomic windows.
    
    :param dict regions: Dictionary of sums corresponding to genomic regions.
    """
    sums = np.array([regions[key]["sums"] for key in regions])
    denoms = np.array([regions[key]["denoms"] for key in regions])
    raw_means = np.zeros(sums.shape[1:], dtype=np.float64)
    raw_means[:-1, :] = sums[:, :-1].sum(0) / denoms[:, :-1].sum(0)[:, None]
    raw_means[-1, :] = sums[:, -1].sum(0) / denoms[:, -1].sum()
    means = [raw_means[i] for i in range(len(raw_means))]
    return means


def weighted_means_across_regions(regions, aggregate=False):
    """
    Compute mean mutation-rate weighted D+ across a dictionary of genomic
    regions.
    
    :param dict regions: Dictionary of sums corresponding to genomic regions.
    :param bool aggregate: If True, compute average uL * uR across the whole
        data set and use it to calculate weights. If False (default), computes
        uL * uR in each region.
    """
    # Construct arrays
    sums = np.array([regions[key]["sums"] for key in regions])
    denoms = np.array([regions[key]["denoms"] for key in regions])
    mut_facs = np.array([regions[key]["mut_facs"] for key in regions])

    if aggregate:
        # Compute genome-wide average ul * ur
        tot_pair_count = denoms[:, :-1].sum()
        tot_ulur = mut_facs[:, :-1].sum()
        avg_ulur = tot_ulur / tot_pair_count
        # Sum ul * ur over the genome but keep bins separate
        factors = mut_facs[:, :-1].sum(0) / avg_ulur
        factors = factors[:, None]
    else:
        # Compute average ul * ur in each interval and bin
        tot_pair_counts = denoms[:, :-1].sum(1)
        tot_ulurs = mut_facs[:, :-1].sum(1)
        avg_ulurs = (tot_ulurs / tot_pair_counts)[:, None]
        # Compose factors for each interval and sum them up
        all_factors = mut_facs[:, :-1] / avg_ulurs
        factors = all_factors.sum(0)
        factors = factors[:, None]

    raw_means = np.zeros(sums.shape[1:], dtype=np.float64)
    raw_means[:-1, :] = sums[:, :-1].sum(0) / factors
    raw_means[-1, :] = sums[:, -1].sum(0) / denoms[:, -1].sum()
    # Convert means to a list
    means = [raw_means[i] for i in range(len(raw_means))]
    return means


def _means_across_replicates(replicates):
    """
    Compute mean statistics across a list of replicates. 
    """
    rep_dict = {i: replicate for i, replicate in enumerate(replicates)}
    return means_across_regions(rep_dict)


def _weighted_means_across_replicates(replicates, aggregate=False):
    """
    Operates on a list of dictionaries; wraps `weighted_means_across_regions`.
    """
    rep_dict = {i: replicate for i, replicate in enumerate(replicates)}
    return weighted_means_across_regions(rep_dict, aggregate=aggregate)
