"""

"""

import demes
import moments
import numpy as np

from . import utils, parsing


class H2stats:
    """
    Class for holding expected H2 and heterozygosity statistics computed with
    moments.LD.

    Parameters
    ----------
    """

    def __init__(self, data, pop_ids):
        self.data = data
        self.pop_ids = pop_ids

    def __getitem__(self, key):
        """
        """
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return self.data[key]
        elif isinstance(key, str):
            if key in self.names[0]:
                idx = self.names[0].index(key)
                return [row[idx] for row in self.data[:-1]]
            elif key in self.names[1]:
                idx = self.names[1].index(key)
                return self.data[-1][idx]
            else:
                raise ValueError("key is unknown")
        else:
            raise ValueError("key type is not supported")

    def __repr__(self):
        return f"H2Stats({self.data}, {self.pop_ids})"

    def __str__(self):
        return f"H2Stats instance: {self.n_pops} pops, {self.n_bins} bins"

    @property
    def n_pops(self):
        return len(self.pop_ids)

    @property
    def n_bins(self):
        return len(self.data) - 1

    @property
    def names(self):
        """Access the names of H2 and H statistics."""
        return [self.h2_names, self.h_names]

    @property
    def flat_names(self):
        return [x for names in self.names for x in names]

    @property
    def h_names(self):
        return parsing._h_names(self.n_pops)

    @property
    def h2_names(self):
        return parsing._h2_names(self.n_pops)

    def H(self, pops=None):
        if pops is not None:
            return self[-1]
        else:
            return self[-1]

    def H2(self, pops=None):
        return

    def h_matrix(self):
        return

    def f2(self, ii, jj):
        return

    def f2_matrix(self):
        return

    def f3(self, ii, jj, kk):
        return

    def f4(self, ii, jj, kk, ll):
        return

    @classmethod
    def from_moments(
        cls,
        graph,
        sampled_demes=None,
        sample_times=None,
        r_bins=None,
        rs=None,
        mu=None,
        theta=None,
        phased=False,
        method="simpsons",
        ):
        """
        Compute expected statistics using moments.LD.
        """
        if isinstance(graph, str):
            graph = demes.load(graph)

        if sampled_demes is None:
            sampled_demes = []

        if sample_times is None:
            pass

        def model_func(_rs):
            """Compute H2 at given recombination distances"""
            ld_stats = moments.Demes.LD(
                graph,
                sampled_demes,
                sample_times=sample_times,
                theta=theta,
                r=_rs,
                u=mu
                )
            n_demes = len(sampled_demes)
            n_stats = int(n_demes * (n_demes + 1) / 2)
            ret = [np.zeros(n_stats, dtype=np.float64)
                   for _ in range(len(_rs))]
            idx = 0
            for ii in range(n_demes):
                for jj in range(ii, n_demes):
                    h2 = ld_stats.H2(ii, jj, phased=phased)
                    for bb, h2_b in enumerate(h2):
                        ret[bb][idx] = h2_b
                    idx += 1
            ret.append(ld_stats.H())
            return ret

        if rs is not None and r_bins is None:
            data = model_func(rs)

        elif rs is None and r_bins is not None:
            if method == "midpoint":
                midpoints = (r_bins[:-1] + r_bins[1:]) / 2
                data = model_func(midpoints)

            elif method == "log_midpoint":
                log_bins = np.log10(r_bins)
                log_mids = 10 ** ((log_bins[:-1] + log_bins[1:]) / 2)

            elif method == "simpsons":
                y_bins = model_func(r_bins)
                midpoints = (r_bins[:-1] + r_bins[1:]) / 2
                y_mids = model_func(midpoints)
                data = [(y_bins[i] + 4 * y_mids[i] + y_bins[i + 1]) / 6
                        for i in range(len(midpoints))]
                # Append heterozygosity statistics
                data.append(y_bins[-1])

            else:
                raise ValueError("unknown approximation method")

        else:
            raise ValueError("either `rs` or `r_bins` must be provided")

        return cls(data, sampled_demes)


class LDstats(moments.LD.LDstats):
    """
    An extension of the moments LDstats 

    Under construction.
    """

    pass


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _sort_name(name):
    """Maps degenerate names H_0_1, H_1_0 to the sorted from H_0_1."""
    split_name = name.split("_")
    symbol = split_name[0]
    idxs = sorted([int(x) for x in split_name[1:]])
    sorted_name = symbol + "_" + "_".join([str(x) for x in idxs])
    return sorted_name

