"""
This file contains a number of smoothers left gauss filter used to smooth a 1-d signal.
Code is courtesy of Addison Hu (minor adjustments by Maria).

Author: Maria Jahja
Created: 2020-04-16

"""
import pytest

import os

import numpy as np
from delphi_utils import (
    left_gauss_linear_smoother,
    moving_window_smoother,
    causal_savgol_smoother,
    impute_with_savgol,
    causal_savgol_coeffs
)


class TestSmoothers:
    def test_moving_window_smoother(self):
        # The raw and smoothed lengths should match
        signal = np.ones(30)
        smoothed_signal = moving_window_smoother(signal)
        assert len(signal) == len(smoothed_signal)

        # The raw and smoothed arrays should be identical on constant data
        # modulo the nans
        signal = np.ones(30)
        window_length = 10
        smoothed_signal = moving_window_smoother(signal, window_length)
        assert np.allclose(
            signal[window_length - 1 :], smoothed_signal[window_length - 1 :]
        )

    def test_left_gauss_linear_smoother(self):
        # The raw and smoothed lengths should match
        signal = np.ones(30)
        smoothed_signal = left_gauss_linear_smoother(signal)
        assert len(signal) == len(smoothed_signal)

        # The raw and smoothed arrays should be identical on constant data
        # modulo the nans
        signal = np.ones(30)
        smoothed_signal = left_gauss_linear_smoother(signal)
        assert np.allclose(signal[1:], smoothed_signal[1:])

        # The raw and smoothed arrays should match when the Gaussian kernel
        # is set to weigh the present value overwhelmingly
        signal = np.arange(1, 10) + np.random.normal(0, 1, 9)
        assert np.allclose(left_gauss_linear_smoother(signal, h=0.1)[1:], signal[1:])

    def test_causal_savgol_coeffs(self):
        # The coefficients should return standard average weights for M=0
        nl, nr = -10, 0
        assert np.allclose(
            causal_savgol_coeffs(nl, nr, 0), np.ones(nr - nl + 1) / (nr - nl + 1)
        )

        # The method should raise an exception when nl >= nr
        nl, nr = 5, 2
        with pytest.raises(ValueError):
            causal_savgol_coeffs(nl, nr, 2)

    def test_causal_savgol_smoother(self):
        # The raw and smoothed lengths should match
        signal = np.ones(30)
        window_length = 10
        smoothed_signal = causal_savgol_smoother(signal, wl=window_length)
        assert len(signal) == len(smoothed_signal)

        # The raw and smoothed arrays should be identical on constant data
        # modulo the nans, when M >= 0
        signal = np.ones(30)
        window_length = 10
        smoothed_signal = causal_savgol_smoother(signal, M=0, wl=window_length)
        assert np.allclose(
            signal[window_length - 1 :], smoothed_signal[window_length - 1 :]
        )

        # The raw and smoothed arrays should be identical on linear data
        # modulo the nans, when M >= 1
        signal = np.arange(30)
        smoothed_signal = causal_savgol_smoother(signal, M=1, wl=window_length)
        assert np.allclose(
            signal[window_length - 1 :], smoothed_signal[window_length - 1 :]
        )

        # The raw and smoothed arrays should be identical on quadratic data
        # modulo the nans, when M >= 2
        signal = np.arange(30) ** 2
        smoothed_signal = causal_savgol_smoother(signal, M=2, wl=window_length)
        assert np.allclose(
            signal[window_length - 1 :], smoothed_signal[window_length - 1 :]
        )

    def test_impute_with_savgol(self):
        # should impute the next value in a linear progression with M>=1
        signal = np.hstack([np.arange(10), [np.nan], np.arange(10)])
        wl = 10
        M = 1
        assert np.allclose(
            impute_with_savgol(signal, wl, M), np.hstack([np.arange(11), np.arange(10)])
        )
        M = 2
        assert np.allclose(
            impute_with_savgol(signal, wl, M), np.hstack([np.arange(11), np.arange(10)])
        )

        # if there are nans on the boundary, should dynamically change window
        signal = np.hstack(
            [np.arange(5), [np.nan], np.arange(20), [np.nan], np.arange(5)]
        )
        wl = 10
        M = 1
        assert np.allclose(
            impute_with_savgol(signal, wl, M),
            np.hstack([np.arange(6), np.arange(21), np.arange(5)]),
        )

        # if the array begins with np.nan, we should tell the user to peel it off before sending
        signal = np.hstack([[np.nan], np.arange(20), [np.nan], np.arange(5)])
        wl = 10
        M = 1
        with pytest.raises(ValueError):
            impute_with_savgol(signal, wl, M)
