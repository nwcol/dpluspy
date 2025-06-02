"""
Houses utilities for reading/writing files and doing arithmetic.
"""

import copy
from datetime import datetime
import gzip
import numpy as np
import re
import pickle
import warnings


## Generating the names of statistics


def _generate_pairs(pop_ids):
    """
    Generate a list of 2-tuples holding the (n choose 2) unique pairs that may
    be drawn from `pop_ids`. 

    :dtype pop_ids: list
    :rtype: list of tuples
    """
    pairs = []
    for i, pop_i in enumerate(pop_ids):
        for pop_j in pop_ids[i:]:
            pairs.append((pop_i, pop_j))
    return pairs


def _get_H_names(pop_ids):
    """
    Generate a list of names of the unique one and two-population H statistics 
    corresponding to a list of population IDs. 

    :dtype pop_ids: list of strings
    :rtype: list of strings
    """
    names = []
    for i, pop_id0 in enumerate(pop_ids):
        for pop_id1 in pop_ids[i:]:
            names.append(f"H_{pop_id0}_{pop_id1}")
    return names


def _get_Dplus_names(pop_ids):
    """
    Get a list of the unique one and two-population D+ statistics corresponding
    to a list of population IDs.

    :dtype pop_ids: list of strings
    :rtype: list of strings
    """
    names = []
    for i, pop_id0 in enumerate(pop_ids):
        for pop_id1 in pop_ids[i:]:
            names.append(f"D+_{pop_id0}_{pop_id1}")
    return names


def _get_stat_names(pop_ids):
    """
    Get the names of all the D+ and H statistics for populations `pop_ids`.
    Statistic names have the form 'D+_{pop_i}_{pop_j}' and 'H_{pop_i}_{pop_j}'.

    :param pop_ids: List of population names.
    :type pop_ids: list of str

    :returns: Lists of names for D+ and H statistics.
    :rtype: tuple of lists of strings
    """
    Dplus_names = _get_Dplus_names(pop_ids)
    H_names = _get_H_names(pop_ids)
    return (Dplus_names, H_names)


def _get_latex_names(pop_ids, statistic="D^+"):
    """
    From a list of population names, get a list of strings of the form 
    '${statistic}_{pop0,pop1}$' for each pair of populations.

    :param pop_ids: List of population names.
    :type pop_ids: list
    :param statistic: Name of the statistic (default 'D^+')
    :type statistic: str

    :returns: A list of string statistic names in a LaTeX-friendly format.
    :rtype: list of strings
    """
    names = []
    for i, pop0 in enumerate(pop_ids):
        for pop1 in pop_ids[i:]:
            if pop0 == pop1:
                names.append(rf"${statistic}_{{\text{{{pop0}}}}}$")
            else:
                names.append(rf"${statistic}_{{\text{{{pop0}, {pop1}}}}}$")
    return names


## BED files and genetic masks


def _read_bed_file(filename):
    """
    Load regions from a BED file as an array. Expects the structure 
        CHROM\tSTART\tEND...\n
    on each line, and skips any comment/header lines that begin with '#'. 
    Raises an error if the BED file has more than one unique CHROM entry.

    :param filename: Pathname of the BED file to load. 
    :type filename: str

    :returns: ndarray of BED file regions, BED chromosome ID
    :rtype: np.ndarray, str
    """
    if filename.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open
    chroms = []
    starts = []
    ends = []
    with openfunc(filename, "rb") as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                continue
            split_line = line.split()
            chroms.append(split_line[0])
            starts.append(float(split_line[1]))
            ends.append(float(split_line[2]))
    chrom_set = set(chroms)
    # check that there is one unique CHROM
    if len(chrom_set) > 1:
        raise ValueError('BED files must describe one chromosome only')
    # check to make sure one or more lines were read
    elif len(chrom_set) == 0:
        raise ValueError('BED file has no valid contents')
    chrom = list(chrom_set)[0]
    regions = np.array(
        [[start, end] for start, end in zip(starts, ends)], dtype=np.int64
    )
    return regions, chrom


def _read_bed_file_positions(bed_file):
    """
    Read a BED file and return a vector of the positions recorded in its
    intervals (0-indexed).
    """
    regions = _read_bed_file(bed_file)[0]
    mask = _regions_to_mask(regions)
    positions = np.nonzero(~mask)[0]
    return positions


def _write_bed_file(filename, regions, chrom):
    """
    Write a BED file. Does not write a header.

    :param filename: Pathname of output file. Should end in .bed or .bed.gz. 
    :type filename: str
    :param regions: Array of BED regions to save.
    :type regions: np.ndarray
    :param chrom: Chromosome number to use in the CHROM column. All regions are 
        assigned to the same chromosome. 
    :type chrom: str

    :returns: None
    """
    if filename.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open
    with openfunc(filename, 'wb') as fout:
        fout.write('#CHROM\tSTART\tEND\n'.encode())
        for start, end in regions:
            fout.write(f'{chrom}\t{start}\t{end}\n'.encode())
    return 


def _regions_to_mask(regions, length=None):
    """
    Return a boolean mask array that equals False within intervals in `regions` 
    and True elsewhere.

    :param regions: Array of intervals.
    :param length: Optional maximum mask length (default None).

    :returns: Boolean mask array.
    """
    if length is None:
        length = regions[-1, 1]
    mask = np.ones(length, dtype=bool)
    for (start, end) in regions:
        if start >= length:
            continue
        elif end > length:
            end = length
        mask[start:end] = 0
    return mask


def _mask_to_regions(mask):
    """
    Return an array of intervals that equal False in a boolean mask array
    (0-indexed).
    """
    jumps = np.diff(np.concatenate(([1], mask, [1])))
    starts = np.where(jumps == -1)[0]
    ends = np.where(jumps == 1)[0]
    regions = np.stack([starts, ends], axis=1)
    return regions


def _intersect_regions(region_arrs):
    """
    Build an array of intervals where every input regions array has coverage.

    :param region_arrs: List of BED region arrays.
    :type region_arrs: list of np.ndarray

    :returns: Region array representing intersection of sites in inputs.
    :rtype: np.ndarray
    """
    max_length = max([regions[-1, 1] for regions in region_arrs])
    # Don't intersect more than 128 masks.
    coverage = np.zeros(max_length, dtype=np.int8)
    for regions in region_arrs:
        for (start, end) in regions:
            coverage[start:end] += 1
    mask = coverage < len(region_arrs)
    intersection = _mask_to_regions(mask)
    return intersection


def _collapse_regions(regions):
    """
    Collapse any overlapping intervals in an array together.
    """
    return _mask_to_regions(_regions_to_mask(regions))


# BEDGRAPH files and recombination maps


def _read_bedgraph_file(filename, override_cols=None, sep=None):
    """
    From a bedgraph-format file, read and return an array of genomic intervals,
    a dictionary of data and the associated chromosome number. There must be
    a header with format corresponding to
        Chrom\tchromStart\tchromEnd\tdata_col1\t...\n
    in the first line. Other commented or header lines beginning with '#' will
    be ignored.

    :param filename: Pathname of the file to load.
    :param sep: File seperator to expect (default None uses whitespace).
    :param override_cols: If given, overrides the data field names in the file 
        header (default None).

    :returns: Array of intervals, dictionary of data arrays, and chromosome ID
    """
    if filename.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open
    chroms = []
    starts = []
    ends = []
    with openfunc(filename, "rb") as fin:
        header_line = fin.readline().decode()
        if header_line[0] != '#':
            raise ValueError('Input file lacks a header line')
        split_header = header_line.replace('#', '').strip().split(sep)
        if override_cols is not None:
            if len(override_cols) != len(split_header) - 3:
                raise ValueError('Invalid `override_cols`')
        raw_data = {i: [] for i in range(3, len(split_header))}
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                continue
            split_line = line.strip().split(sep)
            chroms.append(split_line[0])
            starts.append(split_line[1])
            ends.append(split_line[2])
            for idx in raw_data:
                raw_data[idx].append(split_line[idx])
    chrom_set = set(chroms)
    # check that there is one unique CHROM
    if len(chrom_set) > 1:
        raise ValueError('BED files must describe one chromosome only')
    # check to make sure one or more lines were read
    elif len(chrom_set) == 0:
        raise ValueError('BED file has no valid contents')
    chrom = list(chrom_set)[0]
    data = {}
    for idx in raw_data:
        if override_cols is None:
            field = split_header[idx]
        else:
            field = override_cols[idx - 3]
        if '.' in raw_data[idx][0]:
            arr = np.array(raw_data[idx], dtype=np.float64)
        else:
            arr = np.array(raw_data[idx], dtype=np.int64)
        data[field] = arr
    regions = np.array(
        [[start, end] for start, end in zip(starts, ends)], dtype=np.int64
    )
    return regions, data, chrom


def _write_bedgraph_file(filename, regions, data, chrom_num, sep=None):
    """
    Write a .bedgraph-format file from an array of regions/windows and a 
    dictionary of data columns.
    """
    for field in data:
        if len(data[field]) != len(regions):
            raise ValueError(f'data field {data} mismatches region length!')
    if sep is None:
        sep = '\t'
    if filename.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open
    constants = ['#chrom', 'chromStart', 'chromEnd']
    fields = list(data.keys())
    header = sep.join(constants + fields) + '\n'
    with openfunc(filename, 'wb') as fout:
        fout.write(header.encode())
        for i, (start, end) in enumerate(regions):
            interval = [chrom_num, str(start), str(end)]
            line_data = [str(data[field][i]) for field in fields]
            line = sep.join(interval + line_data) + '\n'
            fout.write(line.encode())
    return


def _read_bedgraph_map(filename, map_col=None, sep=None):
    """
    Read a map from a BEDGRAPH file, returning an array of physical and of map
    coordinates. If no `map_col` is given, accesses the rightmost column.
    Physical coordinates are given as the end points of BEDGRAPH intervals.
    """
    intervals, data, _ = _read_bedgraph_file(filename, sep=sep)
    coords = intervals[:, 1]
    if map_col is None:
        map_col = list(data.keys())[-1]
    map_coords = data[map_col]
    return coords, map_coords


def _read_hapmap_map(filename, pos_col="Position(bp)", map_col="Map(cM)"):
    """
    Read a recombination map in the Hapmap format, returning arrays of physical
    and map coordinates. The first line must be a header.
    """
    if filename.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open 
    coords = []
    map_coords = []
    with openfunc(filename, "rb") as fin:
        header_line = fin.readline().decode()
        if '"' in header_line:
            header_line = header_line.replace('"', '')
        header_line = header_line.split()
        coord_idx = header_line.index(pos_col)
        map_idx = header_line.index(map_col)
        for line in fin:
            split_line = line.decode().strip().split()
            coords.append(split_line[coord_idx])
            map_coords.append(split_line[map_idx])
    coords = np.array(coords, dtype=np.int64)
    map_coords = np.array(map_coords, dtype=np.float64)
    return coords, map_coords


## Recombination map math


def _map_function(r):
    """
    Haldane's map function; transforms distances in recombination distance `r` 
    to Morgans.

    :param r: Float or array of recombination frequencies/distances.
    :type r: float or np.ndarray

    :returns: Distances transformed to Morgans.
    :rtype: float or np.ndarray
    """
    if np.any(r > 0.5):
        raise ValueError('`r` > 0.5 are not allowed')
    if np.any(r < 0):
        raise ValueError('Negative `r` are not allowed')
    return -1 / 2 * np.log(1 - 2 * r)


def _inverse_map_function(d):
    """
    The inverse of Haldane's map function. Transforms distance in Morgans to 
    `r`.

    :param d: Float or array or genetic distances in Morgans.
    :type d: float or np.ndarray

    :returns: Distances transformed to `r`.
    :rtype: float or np.ndarray
    """
    return (1 - np.exp(2 * -d)) / 2


## Printouts


def _current_time():
    """
    Return a string giving the time and date with yyyy-mm-dd format.
    """
    return '[' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']'


## Simulation helper functions


def _increment1(x):
    """
    Increment 1 to every value of x. Used to transform simulated VCF sites to
    1-indexing.
    """
    return [_ + 1 for _ in x]
