"""
This file contains a number of smoothers left gauss filter used to smooth a 1-d signal.
Code is courtesy of Addison Hu (minor adjustments by Maria).

Author: Maria Jahja
Created: 2020-04-16

"""
import pytest

import os

import numpy as np
from delphi_utils import Smoother


class TestSmoothers:
    """
    In all the tests below, X is the signal to be smoothed.
    """
    def test_moving_average_smoother(self):
        # The raw and smoothed lengths should match
        X = np.ones(30)
        smoother = Smoother(method_name='moving_average')
        smoothed_X = smoother.smooth(X)
        assert len(X) == len(smoothed_X)

        # The raw and smoothed arrays should be identical on constant data
        # modulo the nans
        X = np.ones(30)
        window_length = 10
        smoother = Smoother(method_name='moving_average', window_length=window_length)
        smoothed_X = smoother.smooth(X)
        assert np.allclose(
            X[window_length - 1 :], smoothed_X[window_length - 1 :]
        )

    def test_left_gauss_linear_smoother(self):
        # The raw and smoothed lengths should match
        X = np.ones(30)
        smoother = Smoother(method_name='local_linear')
        smoothed_X = smoother.smooth(X)
        assert len(X) == len(smoothed_X)
        # The raw and smoothed arrays should be identical on constant data
        # modulo the nans
        assert np.allclose(X[1:], smoothed_X[1:])

        # The raw and smoothed arrays should match when the Gaussian kernel
        # is set to weigh the present value overwhelmingly
        X = np.arange(1, 10) + np.random.normal(0, 1, 9)
        smoother = Smoother(method_name='local_linear', gaussian_bandwidth=0.1)
        assert np.allclose(smoother.smooth(X)[1:], X[1:])

    def test_causal_savgol_coeffs(self):
        # The coefficients should return standard average weights for M=0
        nl, nr = -10, 0
        window_length = nr - nl + 1
        smoother = Smoother(method_name='savgol', window_length=window_length, poly_fit_degree=0)
        assert np.allclose(
            smoother.coeffs, np.ones(window_length) / window_length
        )

    def test_causal_savgol_smoother(self):
        # The raw and smoothed lengths should match
        X = np.ones(30)
        window_length = 10
        smoother = Smoother(method_name='savgol', window_length=window_length, poly_fit_degree=0)
        smoothed_X = smoother.smooth(X)
        assert len(X) == len(smoothed_X)
        # The raw and smoothed arrays should be identical on constant data
        # modulo the nans, when M >= 0
        assert np.allclose(
            X[window_length - 1 :], smoothed_X[window_length - 1 :]
        )

        # The raw and smoothed arrays should be identical on linear data
        # modulo the nans, when M >= 1
        X = np.arange(30)
        smoother = Smoother(method_name='savgol', window_length=window_length, poly_fit_degree=1)
        smoothed_X = smoother.smooth(X)
        assert np.allclose(
            X[window_length - 1 :], smoothed_X[window_length - 1 :]
        )

        # The raw and smoothed arrays should be identical on quadratic data
        # modulo the nans, when M >= 2
        X = np.arange(30) ** 2
        smoother = Smoother(method_name='savgol', window_length=window_length, poly_fit_degree=2)
        smoothed_X = smoother.smooth(X)
        assert np.allclose(
            X[window_length - 1 :], smoothed_X[window_length - 1 :]
        )

    def test_impute_with_savgol(self):
        # should impute the next value in a linear progression with M>=1
        X = np.hstack([np.arange(10), [np.nan], np.arange(10)])
        window_length = 10
        smoother = Smoother(method_name='savgol', window_length=window_length)
        imputed_X = smoother.impute_with_savgol(X, window_length, poly_fit_degree=1)
        assert np.allclose(
            imputed_X, np.hstack([np.arange(11), np.arange(10)])
        )
        imputed_X = smoother.impute_with_savgol(X, window_length, poly_fit_degree=2)
        assert np.allclose(
            imputed_X, np.hstack([np.arange(11), np.arange(10)])
        )

        # if there are nans on the boundary, should dynamically change window
        X = np.hstack(
            [np.arange(5), [np.nan], np.arange(20), [np.nan], np.arange(5)]
        )
        imputed_X = smoother.impute_with_savgol(X, window_length, poly_fit_degree=2)
        assert np.allclose(
            imputed_X,
            np.hstack([np.arange(6), np.arange(21), np.arange(5)]),
        )

        # if the array begins with np.nan, we should tell the user to peel it off before sending
        X = np.hstack([[np.nan], np.arange(20), [np.nan], np.arange(5)])
        with pytest.raises(ValueError):
            imputed_X = smoother.impute_with_savgol(X, window_length, poly_fit_degree=2)
