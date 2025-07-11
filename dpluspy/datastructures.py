"""
Class for holding expected D+ and H statistics.
"""

import demes
import moments
import numpy as np

from . import utils


class DPlusStats(list):
    """
    A class for holding expected D+ and H statistics.
    """

    def __new__(self, data, pop_ids=None):
        """
        Instantiate from a list of data arrays.

        :param list data: List holding arrays of expected or mean statistics.
        :param list pop_ids: List of population IDs.
        """
        ret = super(DPlusStats, self).__new__(self, data, pop_ids=None)
        if hasattr(data, "pop_ids"):
            ret.num_pops = data.pop_ids
        else:
            ret.num_pops = pop_ids
        return ret

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def DP(self):
        """
        Return D+ statistics, which are stored as a list of arrays. Each array
        corresponds to a recombination distance bin or value.
        """
        return self[:-1]

    def H(self):
        """
        Return H statistics, which are stored in an array.
        """
        return self[-1]

    def names(self):
        """
        Return the full names of the statistics stored in an instance, in the
        format ((D+_{pop0}_{pop0}, ...), (H_{pop0}_{pop0}, ...).

        :returns: 2-tuple of lists holding names of D+, H statistics.
        """
        return utils._stat_names(list(range(len(self.pop_ids))))

    @classmethod
    def from_moments(
        cls,
        graph,
        sampled_demes,
        sample_times=None,
        rs=None,
        u=None,
        phased=False
    ):
        """
        Compute expected statistics given a Demes graph using moments.Demes.LD.

        :param str, demes.Graph graph: Demes graph, or pathname to a a demes-
            format .yaml file.
        :param list sampled_demes: List of demes for which to compute expected 
            statistics.
        :param list sample_times: Optional list of times at which demes are 
            sampled. Sample times default to deme end times.
        :param arr rs: List or array of recombination fractions at which to
            compute expected statistics.
        :param float u: The mutation rate to use for computing expectations. 
        :param phased: If True, compute phased two-population expectations 
            (default False).

        :returns: DPlusStats instance holding expected D+ and H statistics.
        """
        if isinstance(graph, str):
            graph = demes.load(graph)
        y = moments.Demes.LD(
            graph, 
            sampled_demes,
            sample_times=sample_times,
            theta=None, 
            r=rs, 
            u=u
        )
        num_demes = len(sampled_demes)
        num_stats = (num_demes ** 2 + num_demes) // 2 
        stats = [np.zeros(num_stats, dtype=np.float64) for _ in rs]
        pairs = utils._generate_pairs(list(range(num_demes)))
        for i, (j, k) in enumerate(pairs):
            if j == k:
                phasing = True
            else:
                phasing = phased
            stat = y.H2(j, k, phased=phasing)
            for l, x in enumerate(stat):
                stats[l][i] = x
        stats.append(y.H())
        return cls(stats, pop_ids=sampled_demes)
    
    def H_matrix(self):
        """
        Return an array of heterozygosities arranged in a 2d array.
        """
        num_pops = len(self.pop_ids)
        arr = np.zeros((num_pops, num_pops))
        kk = 0
        for ii in range(num_pops):
            for jj in range(num_pops):
                if jj < ii:
                    continue 
                if ii == jj:
                    arr[ii, ii] = self.H()[kk]
                else:
                    arr[ii, jj] = arr[jj, ii] = self.H()[kk]
                kk += 1
        return arr

    def F2(self, X, Y):
        """
        Compute the F2 statistic between populations X and Y.

        :param X: Name (str) or index (int) of population X.
        :param Y: As param X.

        :returns float: Estimate of F2 computed from observed diversity.
        """
        if isinstance(X, str):
            X = self.pop_ids.index(X)
            Y = self.pop_ids.index(Y)
        names = self.names()[1]
        H_X = self.H()[names.index(f"H_{X}_{X}")]
        H_Y = self.H()[names.index(f"H_{Y}_{Y}")]
        H_XY = self.H()[names.index(sort_name(f"H_{X}_{Y}"))]
        return H_XY - (H_X + H_Y) / 2
    
    def F2_matrix(self):
        """
        Compute a matrix of F2 statistics across populations. 
        """
        num_pops = len(self.pop_ids)
        arr = np.zeros((num_pops, num_pops))
        kk = 0
        for ii in range(num_pops):
            for jj in range(num_pops):
                if jj < ii:
                    continue 
                if ii == jj:
                    arr[ii, ii] = 0
                else:
                    arr[ii, jj] = arr[jj, ii] = self.F2(ii, jj)
                kk += 1
        return arr

    def F3(self, X, Y, Z):
        """
        Compute F3(X; Y, Z) from observed heterozygosities.
        """
        if isinstance(X, str):
            X = self.pop_ids.index(X)
            Y = self.pop_ids.index(Y)
            Z = self.pop_ids.index(Z)
        names = self.names()[1]
        H_X = self.H()[names.index(f"H_{X}_{X}")]
        H_XY = self.H()[names.index(sort_name(f"H_{X}_{Y}"))]
        H_XZ = self.H()[names.index(sort_name(f"H_{X}_{Z}"))]
        H_YZ = self.H()[names.index(sort_name(f"H_{Y}_{Z}"))]
        return (H_XY + H_XZ - H_YZ - H_X) / 2
    
    def F4(self, X, Y, Z, W):
        """
        Compute F4(X, Y; Z, W).
        """
        if isinstance(X, str):
            X = self.pop_ids.index(X)
            Y = self.pop_ids.index(Y)
            Z = self.pop_ids.index(Z)
            W = self.pop_ids.index(W)
        names = self.names()[1]
        H_XW = self.H()[names.index(sort_name(f"H_{X}_{W}"))]
        H_YZ = self.H()[names.index(sort_name(f"H_{Y}_{Z}"))]
        H_XZ = self.H()[names.index(sort_name(f"H_{X}_{Z}"))]
        H_YW = self.H()[names.index(sort_name(f"H_{Y}_{W}"))]
        return (H_XW + H_YZ - H_XZ - H_YW) / 2


def sort_name(name):
    """
    There is degeneracy between the symbols "H_0_1" and "H_1_0"; this function
    maps both forms to the same sorted form "H_0_1". 
    """
    split_name = name.split("_")
    symbol = split_name[0]
    ii, jj = sorted([int(x) for x in split_name[1:]])
    sorted_name = f"{symbol}_{ii}_{jj}"
    return sorted_name

