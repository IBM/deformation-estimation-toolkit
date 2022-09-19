import numpy as np
import pytest

import endeform.interpolation.TPS as TPS

# Some knots
Nx, Ny = (4, 5)
N = Nx * Ny
M = 10  #  x and y limit
xx, yy = np.meshgrid(np.linspace(0, M, Nx), np.linspace(0, M, Ny))
X = np.column_stack((xx.flatten(), yy.flatten()))
# subset of knots to use in precomputation
knot_indices = [1, 4, 15, 11, 9, 10]
# 3 sets of values: all 0, all 1, all -2
Y = np.column_stack((np.zeros(N), np.ones(N), -2 * np.ones(N)))
C1 = np.zeros(N + 3)
C1[N] = 1
# these are the expected coefficients for the TPS
C_expected = np.column_stack((np.zeros(N + 3), C1, -2 * C1))

# Some query points
NQ = 10
XQ = np.random.random_sample((NQ, 2)) * M
ZQ = np.repeat(Y[[0], :], NQ, axis=0)


class TestTPS:
    tps = TPS.TPS(alpha=0.1)

    def test_TPS_eval_fails_before_fit(self):
        with pytest.raises(RuntimeError):
            self.tps.eval(XQ)

    def test_TPS_fit_with1D(self):
        self.tps.fit(X, Y[:, 1])
        # np.testing.assert_array_almost_equal_nulp(self.tps.coef_, C1[:,np.newaxis])
        # test with ..._nulp fails when using np.linalg.solve since coeffs that should
        # be 0 end up being, say, 1e-16, and there are roughly 4e18 floating point numbers
        # between 0 and 1e-16, try `np.testing._private.utils.nulp_diff(1e-16,0)` to see that
        np.testing.assert_array_almost_equal(self.tps.coef_, C1[:, np.newaxis])

    def test_TPS_fit_with1Dcolumn(self):
        self.tps.fit(X, Y[:, [1]])
        # np.testing.assert_array_almost_equal_nulp(self.tps.coef_, C1[:,np.newaxis])
        # test with ..._nulp fails when using np.linalg.solve since coeffs that should
        # be 0 end up being, say, 1e-16, and there are roughly 4e18 floating point numbers
        # between 0 and 1e-16, try `np.testing._private.utils.nulp_diff(1e-16,0)` to see that
        np.testing.assert_array_almost_equal(self.tps.coef_, C1[:, np.newaxis])

    def test_1DTPS_eval_1point(self):
        z = self.tps.eval(XQ[0, :])
        np.testing.assert_array_almost_equal_nulp(z, 1)

    def test_TPS_fit_with2D(self):
        self.tps.fit(X, Y)
        # np.testing.assert_array_almost_equal_nulp(self.tps.coef_, C_expected)
        # test with ..._nulp fails when using np.linalg.solve since coeffs that should
        # be 0 end up being, say, 1e-16, and there are roughly 4e18 floating point numbers
        # between 0 and 1e-16, try `np.testing._private.utils.nulp_diff(1e-16,0)` to see that
        np.testing.assert_array_almost_equal(self.tps.coef_, C_expected)

    def test_TPS_eval_1point(self):
        z = self.tps.eval(XQ[0, :])
        np.testing.assert_array_almost_equal_nulp(z, [0, 1, -2])

    def test_TPS_eval(self):
        z = self.tps.eval(XQ)
        np.testing.assert_array_almost_equal(z, ZQ)


class TestTPSPrecompute:
    tps = TPS.TPSprecompute(alpha=0.1)

    def test_TPS_eval_fails_before_fit(self):
        with pytest.raises(RuntimeError):
            self.tps.eval(XQ)

    def test_TPSprecompute_with_knots_only(self):
        # precomputation with nodes only is feasible
        self.tps.precompute_RBF(pre_knots=X)
        # still raises error though when eval is called
        with pytest.raises(RuntimeError):
            self.tps.eval(XQ)

    def test_TPSprecompute_fit_with1D_allknots(self):
        self.tps.fit(Ellipsis, Y[:, 1])
        # np.testing.assert_array_almost_equal_nulp(self.tps.coef_, C1[:,np.newaxis])
        # test with ..._nulp fails when using np.linalg.solve since coeffs that should
        # be 0 end up being, say, 1e-16, and there are roughly 4e18 floating point numbers
        # between 0 and 1e-16, try `np.testing._private.utils.nulp_diff(1e-16,0)` to see that
        np.testing.assert_array_almost_equal(self.tps.coef_, C1[:, np.newaxis])

    def test_TPSprecompute_fit_with1D_someknots(self):
        self.tps.fit(knot_indices, Y[knot_indices, 1])
        # np.testing.assert_array_almost_equal_nulp(self.tps.coef_, C1[:,np.newaxis])
        # test with ..._nulp fails when using np.linalg.solve since coeffs that should
        # be 0 end up being, say, 1e-16, and there are roughly 4e18 floating point numbers
        # between 0 and 1e-16, try `np.testing._private.utils.nulp_diff(1e-16,0)` to see that
        np.testing.assert_array_almost_equal(
            self.tps.coef_,
            np.vstack((C1[knot_indices, np.newaxis], C1[-3:, np.newaxis])),
        )

    def test_TPSprecompute_fit_with1Dcolumn(self):
        self.tps.fit(knot_indices, Y[knot_indices, [1]])
        # np.testing.assert_array_almost_equal_nulp(self.tps.coef_, C1[:,np.newaxis])
        # test with ..._nulp fails when using np.linalg.solve since coeffs that should
        # be 0 end up being, say, 1e-16, and there are roughly 4e18 floating point numbers
        # between 0 and 1e-16, try `np.testing._private.utils.nulp_diff(1e-16,0)` to see that
        np.testing.assert_array_almost_equal(
            self.tps.coef_,
            np.vstack((C1[knot_indices, np.newaxis], C1[-3:, np.newaxis])),
        )

    def test_TPSprecompute_RBF_queries(self):
        # Do it with query points only (using knots from above)
        self.tps.precompute_RBF(pre_query_points=XQ)
        # Once more, with all query points and all knots
        self.tps.precompute_RBF(pre_query_points=XQ, pre_knots=X)
        # And fit again so the remaining tests work
        self.tps.fit(Ellipsis, Y[:, 1])

    def test_1DTPSprecompute_eval_pre_1point(self):
        z = self.tps.eval_pre([0])
        np.testing.assert_array_almost_equal_nulp(z, 1)

    def test_TPSprecompute_fit_with2D_allknots(self):
        self.tps.fit(Ellipsis, Y)
        # np.testing.assert_array_almost_equal_nulp(self.tps.coef_, C_expected)
        # test with ..._nulp fails when using np.linalg.solve since coeffs that should
        # be 0 end up being, say, 1e-16, and there are roughly 4e18 floating point numbers
        # between 0 and 1e-16, try `np.testing._private.utils.nulp_diff(1e-16,0)` to see that
        np.testing.assert_array_almost_equal(self.tps.coef_, C_expected)

    def test_TPSprecompute_fit_with2D_someknots(self):
        self.tps.fit(knot_indices, Y[knot_indices, :])
        np.testing.assert_array_almost_equal(
            self.tps.coef_, np.vstack((C_expected[knot_indices, :], C_expected[-3:, :]))
        )

    def test_TPSprecompute_eval_1point(self):
        z = self.tps.eval(XQ[0, :])
        np.testing.assert_array_almost_equal_nulp(z, [0, 1, -2])

    def test_TPSprecompute_eval(self):
        z = self.tps.eval(XQ)
        np.testing.assert_array_almost_equal(z, ZQ)

    def test_TPSprecompute_eval_pre_1point(self):
        z = self.tps.eval_pre([0])
        np.testing.assert_array_almost_equal(z, [[0, 1, -2]])

    def test_TPS_eval_pre(self):
        z = self.tps.eval_pre()
        np.testing.assert_array_almost_equal(z, ZQ)
