"""
This file contains the smoothing utility functions. We have a number of
possible smoothers to choose from: windowed average, local weighted regression,
and a causal Savitzky-Golay filter.

Code is courtesy of Dmitry Shemetov, Maria Jahja, and Addison Hu.

These smoothers are all functions that take a 1D numpy array and return a smoothed
1D numpy array of the same length (with a few np.nans in the beginning). See the
docstrings for details.
"""

import numpy as np
import pandas as pd


class Smoother:
    """
    This is the smoothing utility class. Reasonable defaults are given for all the
    parameters.

    Parameters
    ----------
    method_name: {'savgol', 'local_linear', 'moving_average'}
        This variable specifies the smoothing method. We have three methods, currently:
        * 'savgol' or a Savtizky-Golay smoother
        * 'local_linear' or a Gaussian-weight linear regression smoother
        * 'moving_average' or a moving window average smoother
    window_length: int
        The length of the averaging window for 'savgol' and 'moving_average'.
        This value is in the units of the data, which in Delphi tends to be days.
        Three weeks appears to be a reasonable smoothing default.
    gaussian_bandwidth: float
        If 'local_linear' is used, this is the width of the Gaussian kernel, in units
        of variance. Analogous to window_length for the other two methods: larger units
        put more weight on the past.
    impute: bool
        If True, will fill nan values before smoothing. Currently uses 'savgol' method
        for imputation.
    minval: float or None
        The smallest value to allow in a signal. If None, there is no smallest value.
        Currently only implemented for 'local_linear'.
    poly_fit_degree: int
        A parameter for the 'savgol' method which sets the degree of the polynomial fit.

    Methods
    ----------
    smooth(signal: np.ndarray) -> np.ndarray
        Takes a 1D signal and returns a smoothed version.
    """

    def __init__(
        self,
        method_name="savgol",
        window_length=21,
        gaussian_bandwidth=100,
        impute=True,
        minval=None,
        poly_fit_degree=3,
        boundary_method="identity",
        savgol_weighted=False,
    ):
        self.method_name = method_name
        self.window_length = window_length
        self.gaussian_bandwidth = gaussian_bandwidth
        self.impute = impute
        self.minval = minval
        self.poly_fit_degree = poly_fit_degree
        self.boundary_method = boundary_method
        self.savgol_weighted = savgol_weighted
        if method_name == "savgol":
            self.coeffs = self.causal_savgol_coeffs(
                -self.window_length + 1,
                0,
                self.poly_fit_degree,
                self.savgol_weighted,
                self.gaussian_bandwidth,
            )
        else:
            self.coeffs = None

        METHODS = {"savgol", "local_linear", "moving_average"}

        if self.method_name not in METHODS:
            raise ValueError("Invalid method name given.")

    def smooth(self, signal):
        if self.impute:
            signal = self.impute_with_savgol(
                signal, self.window_length, self.poly_fit_degree
            )

        if self.method_name == "savgol":
            signal_smoothed = self.pad_and_convolve(
                signal, self.coeffs, self.boundary_method
            )
            return signal_smoothed

        elif self.method_name == "local_linear":
            signal_smoothed = self.left_gauss_linear_smoother(
                signal, self.gaussian_bandwidth, self.impute, self.minval
            )
            return signal_smoothed

        elif self.method_name == "moving_average":
            signal_smoothed = self.moving_average_smoother(signal, self.window_length)
            return signal_smoothed

    def moving_average_smoother(
        self, signal: np.ndarray, window_length=21
    ) -> np.ndarray:
        """
        Compute a moving average on signal.

        Parameters
        ----------
        signal: np.ndarray
            Input array

        Returns
        -------
        signal_smoothed: np.ndarray
            An array with the same length as arr, but the first window_length-1 
            entries are np.nan.
        """
        if not isinstance(window_length, int):
            raise ValueError("k must be int.")

        signal_padded = np.append(np.nan * np.ones(window_length - 1), signal)
        signal_smoothed = (
            np.convolve(signal_padded, np.ones(window_length, dtype=int), mode="valid")
            / window_length
        )

        return signal_smoothed

    def left_gauss_linear_smoother(
        self, signal: np.ndarray, gaussian_bandwidth=10, impute=False, minval=None,
    ) -> np.ndarray:
        """
        Smooth the y-values using a local linear regression with Gaussian weights.

        At each time t, we use the data from times 1, ..., t-dt, weighted
        using the Gaussian kernel, to produce the estimate at time t.

        For math details, see the smoothing_docs folder. TL;DR: the A matrix in the 
        code below is the design matrix of the regression.

        Parameters
        ----------
        signal: np.ndarray
            A 1D signal.
        gaussian_bandwidth: float
            The smoothing bandwidth, in units of variance. Larger units put more
            weight on the past.

        Returns
        ----------
        signal_smoothed: np.ndarray
            A smoothed 1D signal.
        """
        n = len(signal)
        signal_smoothed = np.zeros_like(signal)
        A = np.vstack([np.ones(n), np.arange(n)]).T
        for idx in range(n):
            weights = np.exp(-((np.arange(idx + 1) - idx) ** 2) / gaussian_bandwidth)
            AwA = np.dot(A[: (idx + 1), :].T * weights, A[: (idx + 1), :])
            Awy = np.dot(
                A[: (idx + 1), :].T * weights, signal[: (idx + 1)].reshape(-1, 1)
            )
            try:
                beta = np.linalg.solve(AwA, Awy)
                signal_smoothed[idx] = np.dot(A[: (idx + 1), :], beta)[-1]
            except np.linalg.LinAlgError:
                signal_smoothed[idx] = signal[idx] if impute else np.nan
        if minval is not None:
            signal_smoothed[signal_smoothed <= minval] = minval
        return signal_smoothed

    def causal_savgol_coeffs(
        self, nl, nr, poly_fit_degree, weighted=False, gaussian_bandwidth=10
    ):
        """
        Solves for the Savitzky-Golay coefficients. The coefficients c_i
        give a filter so that
            y = \sum_{i=-{n_l}}^{n_r} c_i x_i
        is the value at 0 (thus the constant term) of the polynomial fit 
        through the points {x_i}. The coefficients are c_i are caluclated as
            c_i =  ((A.T @ A)^(-1) @ (A.T @ e_i))_0
        where A is the design matrix of the polynomial fit and e_i is the standard
        basis vector i. This is currently done via a full inversion, which can be
        optimized.

        Parameters
        ----------
        nl: int
            The left window bound for the polynomial fit.
        nr: int
            The right window bound for the polynomial fit.
        poly_fit_degree: int
            The degree of the polynomial to be fit.

        Returns
        ----------
        coeffs: np.ndarray
            A vector of coefficients of length nl that determines the savgol
            convolution filter.
        """
        if nl >= nr:
            raise ValueError("The left window bound should be less than the right.")
        if nr > 0:
            raise ValueError("The filter is no longer causal.")

        A = np.vstack(
            [np.arange(nl, nr + 1) ** j for j in range(poly_fit_degree + 1)]
        ).T
        # TODO: needs testing
        if weighted:
            weights = np.exp(-((np.arange(nl, nr + 1)) ** 2) / gaussian_bandwidth)
            mat_inverse = np.linalg.inv((A.T * weights) @ A) @ (A.T * weights)
        else:
            mat_inverse = np.linalg.inv(A.T @ A) @ A.T
        window_length = nr - nl + 1
        basis_vector = np.zeros(window_length)
        coeffs = np.zeros(window_length)
        for i in range(window_length):
            basis_vector = np.zeros(window_length)
            basis_vector[i] = 1.0
            coeffs[i] = (mat_inverse @ basis_vector)[0]
        return coeffs

    def pad_and_convolve(self, signal, coeffs, boundary_method="identity"):
        """
        Returns a specific type of convolution of the 1D signal with the 1D signal
        coeffs, respecting boundary effects. That is, the output y is
            signal_smoothed_i = signal_i
            signal_smoothed_i = sum_{j=0}^n coeffs_j signal_{i+j}, if i >= len(coeffs) - 1
        In words, entries close to the left boundary are not smoothed, the window does
        not proceed over the right boundary, and the rest of the values are regular
        convolution.

        Parameters
        ----------
        signal: np.ndarray
            A 1D signal.
        coeffs: np.ndarray
            A 1D signal of smoothing weights.
        boundary_method: {'identity', 'nan'}
            Specifies how to handle the lack data on the left boundary.
            * 'identity' keeps the raw data values
            * 'nan' fills them with nans

        Returns
        ----------
        signal_smoothed: np.ndarray
            A smoothed 1D signal of same length as signal.
        """

        # reverse because np.convolve reverses the second argument
        temp_reversed_coeffs = np.array(list(reversed(coeffs)))
        if boundary_method == "identity":
            signal_padded = np.append(np.nan * np.ones(len(coeffs) - 1), signal)
            signal_smoothed = np.convolve(
                signal_padded, temp_reversed_coeffs, mode="valid"
            )
            for ix in range(len(coeffs)):
                signal_smoothed[ix] = signal[ix]
            return signal_smoothed
        elif boundary_method == "nan":
            signal_padded = np.append(np.nan * np.ones(len(coeffs) - 1), signal)
            signal_smoothed = np.convolve(
                signal_padded, temp_reversed_coeffs, mode="valid"
            )
            return signal_smoothed
        else:
            raise ValueError("Unknown boundary method.")

    def impute_with_savgol(self, signal, window_length, poly_fit_degree):
        """
        This method looks through the signal, finds the nan values, and imputes them
        using an M-degree polynomial fit on the previous window_length data points.
        The boundary cases, i.e. nans within wl of the start of the array
        are imputed with a window length shrunk to the data available.

        Parameters
        ----------
        signal: np.ndarray
            A 1D signal to be imputed.
        window_length: int
            The window length of the filter, i.e. the number of past data points to use
            for the fit.
        poly_fit_degree: int
            The degree of the polynomial used in the fit.

        Returns
        ----------
        signal_imputed: np.ndarray
            An imputed 1D signal.
        """
        signal_imputed = np.copy(signal)
        if np.isnan(signal[0]):
            raise ValueError("The signal should not begin with a nan value.")
        for ix in np.where(np.isnan(signal))[0]:
            if ix < window_length:
                nl, nr = -ix, -1
                coeffs = self.causal_savgol_coeffs(nl, nr, poly_fit_degree)
                signal_imputed[ix] = signal[:ix] @ coeffs
            else:
                nl, nr = -window_length, -1
                coeffs = self.causal_savgol_coeffs(nl, nr, poly_fit_degree)
                signal_imputed[ix] = signal[ix - window_length : ix] @ coeffs
        return signal_imputed


# TODO: this needs a test, probably
def smoothed_values_by_geo_id(
    df: pd.DataFrame, method="savgol", **kwargs
) -> np.ndarray:
    """Computes a smoothed version of the variable 'val' within unique values of 'geo_id'

    Currently uses a local weighted least squares, where the weights are given
    by a Gaussian kernel.

    Parameters
    ----------
    df: pd.DataFrame
        A data frame with columns "geo_id", "timestamp", and "val"
    method: {'savgol', 'local_linear', 'moving_average'}
        A choice of window smoother to use. Check the smoother method definitions
        for specific parameters.

    Returns
    -------
    np.ndarray
        A one-dimensional numpy array containing the smoothed values.
    """
    smoother = Smoother(method, **kwargs)

    df = df.copy()
    df["val_smooth"] = 0
    for geo_id in df["geo_id"].unique():
        signal = df[df["geo_id"] == geo_id]["val"].values
        df.loc[df["geo_id"] == geo_id, "val_smooth"] = smoother.smooth(signal)
    return df["val_smooth"].values
