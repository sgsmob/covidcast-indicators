# Smoothing Methods Documentation

This file is meant to document the decision processes behind the use of different smoothers. We have a number of different smoothers available:

* windowed average
* local linear regression
* causal Savitzky-Golay

It is not clear at this time if any of the methods is universally preferred to the other. This document and the accompanying Jupyter Notebook will document the thought process.

What follows are mathematical descriptions of the methods.

## Moving window average

This is the standard smoother that averages a window of the past (say, two weeks worth of values) to estimate the present day. It is implemented in `moving_window_smoother`.

Given $y_1, \dots, y_n$, the smoothed value at $\hat y_n$ for a window of length $k < n$ is
$$\hat y_n = \frac 1 k \sum_{i=n-k+1}^n y_i.$$

## Gaussian-weighted local linear regression

From [the existing documentation](https://github.com/cmu-delphi/delphi-epidata/blob/main/docs/api/covidcast-signals/ght.md#smoothing):
> For each date, we fit a local linear regression, using a Gaussian kernel, with only data on or before that date. (This is equivalent to using a negative half normal distribution as the kernel.) The bandwidth is chosen such that most of the kernel weight is placed on the preceding seven days. The estimate for the data is the local linear regression's prediction for that date.
It is implemented in `left_gauss_linear`.

What follows is a detailed mathematical description of the method. Suppose $y_1, y_2, \dots, y_n$ is a time series we want to smooth, $\hat y_n$ is the smoothed value at time $n$, and $\mathbf y_k$ denotes a vector of the values from 1 to $k$, both ends inclusive. Suppose that the independent variables are simply the time index $x_i = i$. We model $y$ as follows
$$\mathbf y_k = \mathbf X_k \mathbf \beta_k + \mathbf \epsilon_k,$$
where $\mathbf \beta_k \in \mathbb R^k$ is a vector of coefficients, $\mathbf \epsilon \in \mathbb R^k$ is a vector of noise, and
$$\mathbf X_k = \begin{bmatrix} 1 & 1 \\
1 & 2 \\
\vdots & \vdots \\
1 & k\end{bmatrix}.$$
The minimization problem solved is
$$\hat \beta = \arg \min_{\mathbf \beta} \|\mathbf y - \mathbf X \mathbf \beta\|_2^2,$$
which has classical OLS solution
$$\mathbf {\hat \beta} = (\mathbf X^T \mathbf X)^{-1} (\mathbf X^T \mathbf y).$$
The modified OLS solution for a local regression with weight matrix $\mathbf W$ is
$$\mathbf {\hat \beta} = (\mathbf X^T \mathbf W \mathbf X)^{-1} (\mathbf X^T \mathbf W \mathbf y),$$
where we use the negative half-normal kernel for the weight matrix $\mathbf W$, which is a diagonal matrix with entries given by $\mathbf W_{ii} = K_\sigma(k-i)$. Here we have
$$K_\sigma(x) = \frac{\sqrt 2}{\sigma \sqrt{\pi}} \exp \left(- \frac{x^2}{2 \sigma^2} \right).$$
Interpreting "most kernel weight placed on preceding seven days" to mean that 95% of the weight is on those days, we solve $\int_0^7 K_\sigma(x) dx = .95$ for $\sigma$ and obtain $\sigma \approx 3.5$.

To obtain the smoothed $\hat y_n$, this method forecasts with the regression from the previous $n-1$ data points, i.e.
$$\hat y_n = \langle \mathbf {\hat \beta_{n-1}}, (1, n) \rangle.$$

One issue with our current implementation of the method is that it requires a large, weighted OLS solve for every data point.

## Savitzky-Golay filter

The local linear regression above is one type of [kernel smoother](https://en.wikipedia.org/wiki/Kernel_smoother). Many kernel smoothers can be thought of as a local function fitting. For instance, a window average corresponds to a constant line fit, local regression corresponds to a linear function fit, and the Savitsky-Golay family contains the rest of the polynomial fits.

The [Savitzky-Golay method](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) is a method popularized in analytical chemistry because it preserves the heights and widths of the data. Because the method fits a local polynomial to the data, the method preserves local moments of the data (up to the degree of the polynomial fit). Like many regressions on regularly spaced data, the Savitzky-Golay method has a computationally efficient algorithm available. We implement this method in `causal_savgol`.

For the following, we follow the derivation in *Numerical Recipes* by Press et. al., pg. 768. Suppose that we want to fit the polynomial $a_0 + a_1 i + \dots + a_M i^M$ to the values $(f_{-n_L},\dots, f_{n_R})$. The design matrix for the fitting problem is
$$A_{ij} = i^j, \quad i=-n_L, \dots, n_R, \quad j=0,\dots,M$$
and the normal equations for the vector of $a_j$'s in terms of the vector $f_i$'s
$$a = (\mathbf A^T \cdot \mathbf A)^{-1} \cdot (\mathbf A^T \cdot \mathbf f).$$
The specific forms of the involved matrices are
$$(\mathbf A^T \cdot \mathbf A)_{ij} = \sum_{k=-n_L}^{n_R} A_{ki}A_{kj} = \sum_{k=-n_L}^{n_R} k^{i+j},$$
$$(\mathbf A^T \cdot \mathbf f)_j = \sum_{k=-n_L}^{n_R} A_{kj}f_k = \sum_{k=-n_L}^{n_R} k^j f_k.$$
There are now two crucial observations that allow for a computationally efficient method. First, we are interested in $\hat f_0$, the value of the fitted polynomial at $i=0$, which is the coefficient $a_0$. Therefore, we only need a single row of the inverted matrix, which can be done with an LU decomposition and a single backsolve. Second, the coefficient $a_0$ is linearly dependent on the input data $\mathbf f$, so it can be expressed as a weighted sum
$$\hat f_0 = a_0 = \sum_{i=-{n_L}}^{n_R} c_i f_i$$
where $c_i$ is the component of $a_0$ when $\mathbf f = \mathbf e_i$, i.e.
$$c_i = \left[ (\mathbf A^T \cdot \mathbf A)^{-1} \cdot (\mathbf A^T \cdot   \mathbf e_i) \right]_0 = \sum_{m=0}^M \left[ (\mathbf A^T \cdot \mathbf A)^{-1} \right]_{0m} n^m.$$

Although scipy does have the `savgol_filter` [method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html), it is not causal (i.e. not past dependence). We implement our own, taking inspiration from the scipy implementation.

Note that the SG filter subsumes many standard smoothing methods: $M=0$ corresponds to windowed average and $M=1$ corresponds local linear regression.

It should be noted that the computational advantage only exists for regularly spaced data. If there is missing data, a full OLS will need to be solved. To overcome this, we combine the two methods: if the data in a given window is regularly spaced, the convolution approach is used, otherwise a full matrix inversion is solved.
