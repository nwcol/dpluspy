


import numpy as np
import unittest

import dpluspy





class TestHess(unittest.TestCase):

    def test_(self):
        pass


class TestGrad(unittest.TestCase):

    def test_with_analytic_func(self):
        # f(x, y)
        func = lambda p, x, y, z: \
            np.exp(-((1 - p[0]) ** 2) / 2 - (2 - p[1]) ** 2)
        p0 = np.array([1.001, 1.999])
        args = (None,)
        means = None
        varcovs = None 
        u = dpluspy.uncerts.get_grad(p0, func, args, means, varcovs, delta=0.01)

        bounds = (np.array([1, 1]), np.array([10, 10]))

        u = dpluspy.uncerts.get_grad(p0, func, args, means, varcovs, delta=0.01,
            bounds=bounds)
