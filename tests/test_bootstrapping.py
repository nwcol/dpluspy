"""
Tests for the `bootstrapping` module.
"""

import numpy as np
import unittest

from dpluspy import bootstrapping


class TestSubsetting(unittest.TestCase):

    def test_subset_means(self):
        pop_ids = ['A', 'B', 'C']
        means = [np.arange(6)]
        to_A = bootstrapping.subset_means(means, pop_ids, ['A'])
        self.assertEqual(to_A[0], np.array([0]))
        to_C = bootstrapping.subset_means(means, pop_ids, ['C'])
        self.assertEqual(to_C[0], np.array([5]))
        to_AB = bootstrapping.subset_means(means, pop_ids, ['A', 'B'])
        self.assertTrue(np.all(to_AB[0] == np.array([0, 1, 3])))
        # Flipping the order of `to_pops` should give a different result
        to_AC = bootstrapping.subset_means(means, pop_ids, ['A', 'C'])
        self.assertTrue(np.all(to_AC[0] == np.array([0, 2, 5])))     
        to_CA = bootstrapping.subset_means(means, pop_ids, ['C', 'A'])
        self.assertTrue(np.all(to_CA[0] == np.array([5, 2, 0])))

    def test_subset_varcovs(self):
        pop_ids = ['A', 'B', 'C']
        varcovs = [np.arange(36).reshape((6, 6))]
        to_A = bootstrapping.subset_varcovs(varcovs, pop_ids, ['A'])
        to_A_expected = np.array([[0]])
        self.assertTrue(np.all(to_A == to_A_expected))
        to_C = bootstrapping.subset_varcovs(varcovs, pop_ids, ['C'])
        to_C_expected = np.array([[35]])
        self.assertTrue(np.all(to_C == to_C_expected))
        to_AC = bootstrapping.subset_varcovs(varcovs, pop_ids, ['A', 'C'])
        to_AC_expected = np.array([[0, 2, 5], [12, 14, 17], [30, 32, 35]])
        self.assertTrue(np.all(to_AC == to_AC_expected))
        to_CA = bootstrapping.subset_varcovs(varcovs, pop_ids, ['C', 'A'])
        to_CA_expected = np.array([[35, 32, 30], [17, 14, 12], [5, 2, 0]])
        self.assertTrue(np.all(to_CA == to_CA_expected))

    def test_subset_stats_by_bin(self):
        pass


class TestMeansAcrossRegions(unittest.TestCase):

    def test_means_across_regions(self):
        # Deterministic tests
        pass

    def test_means_across_replicates(self):
        pass


class TestBootstrap(unittest.TestCase):

    def test_bootstrap(self):
        pass

    def test_bootstrap_with_sampling(self):
        # Construct a dataset by random sampling
        pass
    

class TestMeansAcrossRegions(unittest.TestCase):

    def test_weighted_means_across_regions(self):
        pass


class TestWeightedBootstrap(unittest.TestCase):

    def test_weighted_bootstrap(self):
        pass


