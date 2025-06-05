"""
Houses a class for holding expected D+ and H statistics.
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

    def Dplus(self):
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
    
    @property
    def stat_names(self):
        """
        Return the full names of the statistics stored in an instance, in the
        format ((D+_{pop0}_{pop0}, ...), (H_{pop0}_{pop0}, ...).

        :returns: 2-tuple of lists holding names of D+, H statistics.
        """
        return utils._get_stat_names(self.pop_ids)

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
    
