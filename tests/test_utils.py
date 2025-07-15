"""
Tests for the `utils` module: mostly concerned with map/mask handling.
"""

import numpy as np
import unittest

from dpluspy import utils 


class TestMaskManips(unittest.TestCase):

    def write_me(self):
        pass


class TestBedFiles(unittest.TestCase):

    def write_me(self):
        pass


class TestBedGraphFiles(unittest.TestCase):

    def write_me(self):
        pass


class TestMapFunctions(unittest.TestCase):

    def test_map_functions(self):
        # Map functions should be inverses
        self.assertAlmostEqual(
            utils._map_function(utils._inverse_map_function(1e-8)), 1e-8)
        self.assertAlmostEqual(
            utils._map_function(utils._inverse_map_function(1e-2)), 1e-2)
        self.assertAlmostEqual(
            utils._map_function(utils._inverse_map_function(1)), 1)
        self.assertAlmostEqual(
            utils._inverse_map_function(utils._map_function(1e-8)), 1e-8)
        self.assertAlmostEqual(
            utils._inverse_map_function(utils._map_function(1e-2)), 1e-2)
        self.assertAlmostEqual(
            utils._inverse_map_function(utils._map_function(0.45)), 0.45)


