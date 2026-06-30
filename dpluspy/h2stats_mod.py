"""

"""

import demes
import moments
import numpy as np


class H2Stats:
    """
    Class for holding expected H2 and heterozygosity statistics computed with
    moments.LD.
    """

    def __init__(self, stats, pops):
        self.stats = stats
        self.pops = pops

