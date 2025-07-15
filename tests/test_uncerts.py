
import numpy as np
import unittest
import dpluspy


class TestHess(unittest.TestCase):

    def setUp(self):
        """
        Evaluates second partial derivatives using the hessian_elem function.
        The following analytical function is used to test accuracy.

        f(x, y) = 2x + x^2 * y^3 + y
        dxdx f = 2y^3
        dxdy f = 6x * y^2
        dydy f = 6x^2 * y
        """
        def func(p, *args):
            x, y = p
            return 2 * x + x ** 2 * y ** 3 + y
        self.func = func

        def func_dxdx(p):
            x, y = p
            return 2 * y ** 3
        self.dxdx = func_dxdx
        
        def func_dxdy(p):
            x, y = p
            return  6 * x * y ** 2
        self.dxdy = func_dxdy
        
        def func_dydy(p):
            x, y = p
            return 6 * x ** 2 * y 
        self.dydy = func_dydy
        
        self.args = (None,)

    def test_hess_elem(self):
        p = np.array([1, 1])
        steps = p * 0.01
        bounds = (np.array([0, 0]), np.array([np.inf, np.inf]))

        H00 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 0, args=self.args)
        analytic_H00 = self.dxdx(p)
        self.assertTrue(np.isclose(H00, analytic_H00, rtol=1e-6))

        H11 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 1, 1, args=self.args)
        analytic_H11 = self.dydy(p)
        self.assertTrue(np.isclose(H11, analytic_H11, rtol=1e-6))
        
        H01 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args)
        H01 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 1, 0, args=self.args)
        analytic_H01 = analytic_H10 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H10, rtol=1e-4))
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=1e-4))

    def test_univariate_bounds(self):
        # x bounded below at 1
        def wfunc(p, *args):
            x, y = p
            if x <= 1:
                return np.nan
            return 2 * x + x ** 2 * y ** 3 + y
        p = np.array([1.01, 1])
        steps = p * 0.01
        bounds = (np.array([1, 0]), np.array([np.inf, np.inf]))
        H00, form = dpluspy.uncerts.hessian_elem(
            wfunc, p, steps, bounds, 0, 0, args=self.args, return_form=True)
        analytic_H00 = self.dxdx(p)
        self.assertTrue(np.isclose(H00, analytic_H00, rtol=1e-6))
        self.assertEqual(form, "forward")

        # x bounded above at 1
        def wfunc(p, *args):
            x, y = p
            if x >= 1:
                return np.nan
            return 2 * x + x ** 2 * y ** 3 + y
        p = np.array([0.995, 1])
        steps = p * 0.01
        bounds = (np.array([0, 0]), np.array([1, np.inf]))
        H00, form = dpluspy.uncerts.hessian_elem(
            wfunc, p, steps, bounds, 0, 0, args=self.args, return_form=True)
        analytic_H00 = self.dxdx(p)
        self.assertTrue(np.isclose(H00, analytic_H00, rtol=1e-6))
        self.assertEqual(form, "backward")

        # y bounded above at 1
        def wfunc(p, *args):
            x, y = p
            if y >= 1:
                return np.nan
            return 2 * x + x ** 2 * y ** 3 + y
        p = np.array([1, 0.999])
        steps = p * 0.01
        bounds = (np.array([0, 0]), np.array([np.inf, 1]))
        H11, form = dpluspy.uncerts.hessian_elem(
            wfunc, p, steps, bounds, 1, 1, args=self.args, return_form=True)
        analytic_H11 = self.dydy(p)
        # This one isn't very precise
        self.assertTrue(np.isclose(H11, analytic_H11, rtol=1e-2))
        self.assertEqual(form, "backward")
        
    def test_univariate_zero_bounds(self):
        # We should be able to evaluate at 0
        p = np.array([0, 1])
        # Default step size for parameter values of 0
        steps = np.array([0.01, 0.01])
        bounds = (np.array([0, 0]), np.array([np.inf, np.inf]))

        H00, form = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 0, args=self.args, return_form=True)
        analytic_H00 = self.dxdx(p)
        self.assertTrue(np.isclose(H00, analytic_H00, rtol=1e-6))
        self.assertEqual(form, "forward")

        p = np.array([1, 0])
        H11, form = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 1, 1, args=self.args, return_form=True)
        analytic_H11 = self.dydy(p)
        # Analytic solution is zero; the estimate is relatively inaccurate
        self.assertTrue(np.isclose(H11, analytic_H11, atol=0.1))
        self.assertEqual(form, "forward")

        p = np.array([0, 0])
        H11, form = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 1, 1, args=self.args, return_form=True)
        analytic_H11 = self.dydy(p)
        # Analytic solutuon is zero
        self.assertTrue(np.isclose(H11, analytic_H11, atol=1e-6))
        self.assertEqual(form, "forward")

    def test_bivariate_bounds(self):
        p = np.array([1, 1])
        steps = p * 0.01

        # backward, backward
        bounds = (np.array([0, 0]), np.array([1.005, 1.005]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=0.05))
        self.assertEqual(form0, "backward,backward")

        # backward, forward
        bounds = (np.array([0, 0.995]), np.array([1.005, np.inf]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=0.05))
        self.assertEqual(form0, "backward,forward")

        # backward, central
        bounds = (np.array([0, 0]), np.array([1.005, np.inf]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=0.05))
        self.assertEqual(form0, "backward,central")

        # forward, backward
        bounds = (np.array([0.995, 0]), np.array([np.inf, 1.005]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=0.01))
        self.assertEqual(form0, "forward,backward")

        # forward, forward
        bounds = (np.array([0.995, 0.995]), np.array([np.inf, np.inf]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=0.05))
        self.assertEqual(form0, "forward,forward")

        # forward, central
        bounds = (np.array([0.995, 0]), np.array([np.inf, np.inf]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=0.05))
        self.assertEqual(form0, "forward,central")

        # central, backward
        bounds = (np.array([0, 0]), np.array([np.inf, 1.005]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=0.05))
        self.assertEqual(form0, "central,backward")

        # central, forward
        bounds = (np.array([0, 0.995]), np.array([np.inf, np.inf]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, rtol=0.05))
        self.assertEqual(form0, "central,forward")

    def test_bivariate_zero_bounds(self):
        p = np.array([0, 0])
        steps = np.array([0.01, 0.01])
        bounds = (np.array([0, 0]), np.array([np.inf, np.inf]))
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        H10, form1 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 1, 0, args=self.args, return_form=True)
        analytic_H01 = analytic_H10 = self.dxdy(p)
        # Analytic solution is zero
        self.assertTrue(np.isclose(H10, analytic_H10, atol=1e-6))
        self.assertEqual(form0, "forward,forward")
        self.assertTrue(np.isclose(H01, analytic_H01, atol=1e-6))
        self.assertEqual(form1, "forward,forward")

        p = np.array([1, 0])
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, atol=1e-3))
        self.assertEqual(form0, "central,forward")

        p = np.array([0, 1])
        H01, form0 = dpluspy.uncerts.hessian_elem(
            self.func, p, steps, bounds, 0, 1, args=self.args, return_form=True)
        analytic_H01 = self.dxdy(p)
        self.assertTrue(np.isclose(H01, analytic_H01, atol=0.05))
        self.assertEqual(form0, "forward,central")


class TestVarMatrix(unittest.TestCase):

    pass


class TestScore(unittest.TestCase):
    """
    Tests evaluation of the gradient, especially at bounds, using an analytic
    function of three variables.
    """
    def setUp(self):
        def func(p, *args):
            x, y, z = p
            return 2 * x + x * y + y ** 2 + np.sqrt(z)
        self.func = func

        def grad(p):
            x, y, z = p
            v = np.array([[2 + y, x + 2 * y, 0.5 * z ** -0.5]]).T
            return v
        self.grad = grad

    def test_score(self):
        p = np.array([1, 1, 1])
        args = (None,)
        ms = None
        vcs = None

        # Use a few different deltas
        u = dpluspy.uncerts.get_score(p, self.func, args, ms, vcs, delta=0.01)
        u_analytic = self.grad(p)
        self.assertTrue(np.all(np.isclose(u, u_analytic, rtol=1e-4)))
        u = dpluspy.uncerts.get_score(p, self.func, args, ms, vcs, delta=0.001)
        self.assertTrue(np.all(np.isclose(u, u_analytic, rtol=1e-4)))
        u = dpluspy.uncerts.get_score(p, self.func, args, ms, vcs, delta=0.05)
        self.assertTrue(np.all(np.isclose(u, u_analytic, rtol=1e-3)))

        # Evaluate at a different point
        p = np.array([100, 0.1, 0.01])
        u = dpluspy.uncerts.get_score(p, self.func, args, ms, vcs, delta=0.01)
        u_analytic = self.grad(p)
        self.assertTrue(np.all(np.isclose(u, u_analytic, rtol=1e-4)))

    def test_lower_bounds(self):
        def func(p, *args):
            x, y, z = p
            return 2 * x + y ** 2 + z
        p0 = np.array([1000, 10, 0.001])
        args = (None,)
        ms = None
        vcs = None
        bounds = (np.array([999, 9.999, 0.000999]), np.array([np.inf] * 3))
        u_analytic = np.array([[2, 2 * p0[1], 1]]).T
        u = dpluspy.uncerts.get_score(
            p0, func, args, ms, vcs, delta=0.01, bounds=bounds)
        self.assertTrue(np.all(np.isclose(u, u_analytic, rtol=0.01)))

        # Parameters are allowed to equal 0
        p0 = np.array([0, 0, 0])
        u_analytic = np.array([[2, 2 * p0[1], 1]]).T
        u = dpluspy.uncerts.get_score(p0, func, args, ms, vcs, delta=0.01)
        self.assertTrue(np.all(np.isclose(u, u_analytic, rtol=0.01, atol=0.01)))

        # Explicitly setting a zero bound should give the same evaluation
        bounds = (np.array([0] * 3), np.array([np.inf] * 3))
        _u = dpluspy.uncerts.get_score(p0, func, args, ms, vcs, delta=0.01)
        self.assertTrue(np.all(_u == u))

    def test_upper_bounds(self):
        def func(p, *args):
            x, y, z = p
            return 2 * x + y ** 2 + z
        p0 = np.array([1000, 10, 0.001])
        args = (None,)
        ms = None
        vcs = None
        bounds = (np.array([0] * 3), np.array([1001, 10.01, 0.001001]))
        u_analytic = np.array([[2, 2 * p0[1], 1]]).T
        u = dpluspy.uncerts.get_score(p0, func, args, ms, vcs, delta=0.01)
        self.assertTrue(np.all(np.isclose(u, u_analytic, rtol=0.01)))

    def test_mixed_bounds(self):
        def func(p, *args):
            x, y, z = p
            return 2 * x + y ** 2 + z
        p0 = np.array([1000, 10, 0.001])
        args = (None,)
        ms = None
        vcs = None
        bounds = (np.array([999.9, 9.999, 0]), np.array([np.inf, 100, 0.001001]))
        u_analytic = np.array([[2, 2 * p0[1], 1]]).T
        u = dpluspy.uncerts.get_score(
            p0, func, args, ms, vcs, delta=0.01, bounds=bounds)
        self.assertTrue(np.all(np.isclose(u, u_analytic, rtol=0.01)))
