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

# from .config import Config

SMOOTHER_BANDWIDTH = 10


def moving_window_smoother(x: np.ndarray, k=7) -> np.ndarray:
    """
    Compute k-day moving average on x.

    Parameters
    ----------
    x: np.ndarray
        Input array

    Returns
    -------
    np.ndarray:
        An array with the same length as arr, but the first k-1 entries are np.nan.
    """
    if not isinstance(k, int):
        raise ValueError("k must be int.")
    x_padded = np.append(np.nan * np.ones(k - 1), x)
    x_smoothed = np.convolve(x_padded, np.ones(k, dtype=int), mode="valid") / k
    return x_smoothed


def left_gauss_linear_smoother(
    x: np.ndarray, h=SMOOTHER_BANDWIDTH, impute=False, minval=None
) -> np.ndarray:
    """
    Smooth the y-values using a local linear regression with Gaussian weights.

    At each time t, we use the data from times 1, ..., t-dt, weighted
    using the Gaussian kernel, to produce the estimate at time t.

    For math details, see the smoothing_docs folder.

    Parameters
    ----------
    x: np.ndarray
        A 1D signal.
    h: float
        The smoothing bandwidth, in units of variance. Larger units put more
        weight on the past.

    Returns
    ----------
    x_smoothed: np.ndarray
        A smoothed 1D signal.
    """
    n = len(x)
    x_smoothed = np.zeros_like(x)
    X = np.vstack([np.ones(n), np.arange(n)]).T
    for idx in range(n):
        wts = np.exp(-((np.arange(idx + 1) - idx) ** 2) / h)
        XwX = np.dot(X[: (idx + 1), :].T * wts, X[: (idx + 1), :])
        Xwy = np.dot(X[: (idx + 1), :].T * wts, x[: (idx + 1)].reshape(-1, 1))
        try:
            beta = np.linalg.solve(XwX, Xwy)
            x_smoothed[idx] = np.dot(X[: (idx + 1), :], beta)[-1]
        except np.linalg.LinAlgError:
            x_smoothed[idx] = x[idx] if impute else np.nan
    if minval is not None:
        x_smoothed[x_smoothed <= minval] = minval
    return x_smoothed


def causal_savgol_coeffs(nl, nr, M):
    """
    Solves for the Savitzky-Golay coefficients. The coefficients c_i
    give a filter such that for data points x_{-n_l}, x_{-n_l+1}, ..., x_{n_r}
    we can find the value at f(0) of the degree-M polynomial f(t) fit to the data.
    The coefficients are c_i are caluclated as
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
    M: int
        The degree of the polynomial to be fit.

    Returns
    ----------
    c: np.ndarray
        A vector of coefficients of length nl that determines the savgol
        convolution filter.
    """
    if nl >= nr:
        raise ValueError("The left window bound should be less than the right.")
    if nr > 0:
        raise ValueError("The filter is no longer causal.")

    A = np.vstack([np.arange(nl, nr + 1) ** j for j in range(M + 1)]).T
    mat_inverse = np.linalg.inv(A.T @ A) @ A.T
    wl = nr - nl + 1
    e = np.zeros(wl)
    c = np.zeros(wl)
    for i in range(wl):
        e = np.zeros(wl)
        e[i] = 1.0
        c[i] = (mat_inverse @ e)[0]
    return c


def causal_savgol_smoother(x, wl=14, M=3):
    """
    Filters a time series x with a Savitzky-Golay filter with window length
    nl and polynomial degree M. The boundaries of the convolution require
    some attention -- currently, they are ignored, but in practice, we will
    want to pad the input data in some way.

    Parameters
    ----------
    x: np.ndarray
        A 1D signal.
    wl: int
        The window length for the polynomial fit. In other words, the number
        of data points for the polynomial fit.
    M: int
        The degree of the polynomial to be fit.

    Returns
    ----------
    x_smoothed: np.ndarray
        A smoothed 1D signal of same length as x.
    """
    c = causal_savgol_coeffs(-wl + 1, 0, M)
    c = np.array(list(reversed(c)))  # reverse because np.convolve reverses
    x_padded = np.append(np.nan * np.ones(len(c) - 1), x)
    x_smoothed = np.convolve(x_padded, c, mode="valid")
    return x_smoothed


def impute_with_savgol(signal, wl, M):
    """
    This method looks through the signal, finds the nan values, and imputes them
    using an M-degree polynomial fit on the previous wl-many data points (wl stands
    for window length). The boundary cases, i.e. nans within wl of the start of the array
    are imputed with a window length shrunk to the data available.

    Parameters
    ----------
    signal: np.ndarray
        A 1D signal to be imputed.
    wl: int
        The window length of the filter, i.e. the number of past data points to use
        for the fit.
    M: int
        The degree of the polynomial used in the fit.

    Returns
    ----------
    imputed_signal: np.ndarray
        An imputed 1D signal.
    """
    imputed_signal = np.copy(signal)
    if np.isnan(signal[0]):
        raise ValueError("The signal should not begin with a nan value.")
    for ix in np.where(np.isnan(signal))[0]:
        if ix < wl:
            nl, nr = -ix, -1
            c = causal_savgol_coeffs(nl, nr, M)
            imputed_signal[ix] = signal[:ix] @ c
        else:
            nl, nr = -wl, -1
            c = causal_savgol_coeffs(nl, nr, M)
            imputed_signal[ix] = signal[ix - wl : ix] @ c
    return imputed_signal


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
    method: {'savgol', 'local_linear', 'window_average'}
        A choice of window smoother to use. Check the smoother method definitions
        for specific parameters.

    Returns
    -------
    np.ndarray
        A one-dimensional numpy array containing the smoothed values.
    """
    METHODS = {
        "savgol": causal_savgol_smoother,
        "local_linear": left_gauss_linear_smoother,
        "window_average": moving_window_smoother,
    }

    if method in METHODS:
        method_function = METHODS[method]
    else:
        raise ValueError("Invalid method name given.")

    df = df.copy()
    df["val_smooth"] = 0
    for geo_id in df["geo_id"].unique():
        x = df[df["geo_id"] == geo_id]["val"].values
        df.loc[df["geo_id"] == geo_id, "val_smooth"] = method_function(x, **kwargs)
    return df["val_smooth"].values

