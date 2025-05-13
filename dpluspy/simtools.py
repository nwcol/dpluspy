"""
Utility functions for setting up coalescent and forward-in-time simulations
"""

import gzip
import msprime
import numpy as np
import scipy

from dpluspy import utils


def build_map(shape, scale, mean_interval, L):
    """
    Build a map (probably a mutation or recombination map) by sampling interval
    lengths from a geometric distribution up to length `L` and assigning each
    interval a Gamma(shape, scale) rate.
    
    :param shape: Shape of the Gamma distribution to sample map rates from.
    :param scale: Scale parameter of the Gamma distribution to sample map rates
        from. 
    :param window_size: Mean window size; windows are drawn from a geometric
        distribution with rate equal to the inverse. 
    :param L: Length of map to construct. 
    :returns: Array of sequence intervals, array of map values
    """
    rate = 1 / mean_interval
    intervals = []
    end = 0
    reached_end = False
    while not reached_end:
        start = end
        end = start + np.random.geometric(rate)
        if end >= L:
            reached_end = True
            end = L
        intervals.append([start, end])
    intervals = np.asarray(intervals, dtype=np.int64)
    num_intervals = len(intervals)
    values = np.random.gamma(shape, scale=scale, size=num_intervals)

    return intervals, values


def write_random_map(filename, shape, scale, mean_interval, L):
    """
    Generate a random genomic map and save it as a .bedgraph file.
    
    :param filename: Pathname for output file.
    See `build_map` for other parameters:
    :returns: None
    """
    intervals, values = build_map(shape, scale, mean_interval, L)
    data = {'map': values}
    utils._write_bedgraph_file(filename, intervals, data, '0')

    return


def load_ratemap(filename):
    """
    Load a ratemap specified in a BEDGRAPH file and use it to construct an 
    msprime.RateMap object. 
    
    """
    # right now: has to have exact sequence length
    intervals, data, _ = utils._read_bedgraph_file(filename)
    positions = np.append(intervals[:, 0], intervals[-1, 1])
    rate = data['map']
    ratemap = msprime.RateMap(position=positions, rate=rate)

    return ratemap



## Old functions for creating synthetic vcf files


def write_random_vcf(
    file, 
    positions,
    sample_ids,
    prob_polymorphism=0.001,
    phased=True
):
    """

    """
    draws = np.random.uniform(size=len(positions))
    sites = positions[draws < prob_polymorphism]

    haplotypes = build_random_haplotypes(len(sites), len(sample_ids))
    genotypes = haplotypes.reshape((len(sites), len(sample_ids), 2))
    
    write_vcf_file(
        file,
        sites,
        genotypes,
        sample_ids,
        phased=phased
    )

    return


def build_random_haplotypes(num_sites, num_samples):
    """
    Construct an array of haplotypes by sampling derived allele counts from
    a neutral SFS and randomly assigning them to haploid samples.
    """
    num_copies = 2 * num_samples

    sfs_raw = np.array([1 / i for i in range(1, num_copies)])
    sfs = sfs_raw / sfs_raw.sum()
    nums_derived = np.random.choice(
        np.arange(1, num_copies), p=sfs, size=num_sites
    )

    count_cache = {}

    for num_derived in range(1, num_copies):
        count_cache[num_derived] = np.hstack(
            (
                np.zeros(num_copies - num_derived, dtype=np.int64),
                np.ones(num_derived, dtype=np.int64)
            )
        )

    haplotypes = np.zeros((num_sites, num_copies), dtype=np.int64)

    for i, num in enumerate(nums_derived):
        haplotypes[i] = np.random.permutation(count_cache[num])
    
    return haplotypes


def write_vcf_file(
    file,
    positions,
    genotypes,
    sample_ids,
    refs=None,
    alts=None,
    phased=True,
    chrom='0'
):
    """
    From arrays of positions and genotypes, write a .vcf with only FORMAT/GT.
    """ 
    open_func = gzip.open if file.endswith('.gz') else open

    chrom = str(chrom).encode()
    bpositions = [str(x).encode() for x in positions]
    sep = b"|" if phased else "/"
    bgenotypes = []
    for gt in genotypes:
        bgt = b"\t".join(
            [str(x[0]).encode() + sep + str(x[1]).encode() for x in gt]
        )
        bgenotypes.append(bgt)

    if refs is None:
        refs = [b"0"] * len(positions)

    if alts is None:
        alts = []
        for gts in genotypes:
            unique = set(list(gts.flatten()))
            unique.remove(0)
            alts.append(b",".join([str(x).encode() for x in list(unique)]))

    with open_func(file, "wb") as fout:
        fout.write(
            b"##fileformat=VCFv4.1\n"
            + b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(sample_ids).encode() + b"\n"
        )
        for i, (pos, gts) in enumerate(zip(bpositions, bgenotypes)):
            line = b"\t".join(
                [
                    chrom,
                    pos, 
                    b".",
                    refs[i],
                    alts[i],
                    b".",
                    b".",
                    b".",
                    b"GT",
                    gts
                ]
            ) + b"\n"

            fout.write(line)

    return



## for coalescent simulations


def discretize_mut_map(mut_map, mask, windows):
    
    tot_mean = np.nanmean(mut_map)
    print(tot_mean)
    if len(mut_map) > len(mask):
        mut_map = mut_map[:len(mask)]
    
    mut_map = np.ma.array(mut_map, mask=mask)

    discrete_mut = np.zeros(len(windows), dtype=np.float64)

    for i, (start, end) in enumerate(windows):
        segment = mut_map[start:end]
        if np.all(segment.mask) or np.all(np.isnan(segment)):
            discrete_mut[i] = tot_mean

        else:
            discrete_mut[i] = np.nanmean(segment)

    print(f"mean segment rate: {np.mean(discrete_mut)}")
    return discrete_mut
