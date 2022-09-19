from warnings import warn
import numpy as np


def find_lower_edge_ind(X, xgrid):
    """Find the left index of the interval in `xgrid` containing each value in `X`.
    Specifically returns `Xind`:
        Xind[i1,i2,...,in] = r
    such that (let `x=X[i1,i2,...,in]`)
        xgrid[r] <= x < xgrid[r+1] if  xgrid.min()<= x <= xgrid.max()
    and
        r=0 if x <= xgrid.min()
        r=len(xgrid)-1 if x>= xgrid.max()
    Parameters
    ----------
    X : np.array
      n-D array of shape (d1, d2, ..., dn)
    xgrid : np.array
        1-D array of length N>1. Needs to be _SORTED in ASCENDING order_

    Returns
    -------
    np.array
        Array of same shape as `X`
    """
    Xind = np.searchsorted(xgrid[1:-1], X)
    return Xind


#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
def bilin_interp(Y, X, ygrid, xgrid, Vgrid, extrapolation_value=None, extrapolation_warning=False):
    """
    Performs bilinear interpolation of data given on a _rectangular_ grid
    `xgrid, ygrid` (both vectors) with the data `V`.

    We adopt the convention that the left-top corner is (0,0), the first index is y, the second index is x.
    Hence, V[y, x,... ] is y pixels down and x pixels to the right of the left top corner.

    Parameters
    ----------
    X,Y : np.arrays of shape (Nx, Ny)
        The coordinates of the points on which the data needs to be interpolated
    xgrid, ygrid : np.arrays of shape (nx,) and (ny,)
        The vectors specifying the grid. Since the grid has to be rectangular, the grid points are `(ygrid[i], xgrid[j])` for all `i` and `j`.
    Vgrid : array of shape (ny, nx, n_channels)
        The data given on the above grid
    extrapolation_value : None or scalar or np.array of shape (n_channels)
        If not None, then extrapolated values (i.e. where (X[i,j], Y[i,j]) falls outside the grid) are filled
        with this value instead of extrapolating.  If scalar, then all channels of `V` are filled with the values.
    extrapolation_warning : bool, optional
        If True, generate a warning when extrapolating. By default False.

    Returns
    --------
    V : np.array of shape (Ny, Nx, n_channels)
        Interpolated data so that V[ii,jj,...] is the interpolation at point '(X[ii,jj],Y[ii,jj])`
    """

    dx = np.diff(xgrid)
    dy = np.diff(ygrid)

    if extrapolation_warning and (
        np.any(X < np.min(xgrid))
        or np.any(Y < np.min(ygrid))
        or np.any(Y > np.max(ygrid))
        or np.any(X > np.max(xgrid))
    ):
        # if we are extrapolating, i.e. there's points outside the grid, warn
        warn("Extrapolating at least one point", RuntimeWarning)

    Xind = find_lower_edge_ind(X, xgrid)
    Yind = find_lower_edge_ind(Y, ygrid)

    if Vgrid.ndim == 3:
        n_chan = Vgrid.shape[2]
        V = np.zeros(X.shape + (n_chan,))
    else:  #  for broadcasting purposes
        n_chan = 1
        V = np.zeros(X.shape)

    f11 = Vgrid[Yind, Xind, ...]
    f12 = Vgrid[Yind, Xind + 1, ...]
    f21 = Vgrid[Yind + 1, Xind, ...]
    f22 = Vgrid[Yind + 1, Xind + 1, ...]

    x2 = (xgrid[Xind + 1] - X)[:, :, np.newaxis]
    x1 = (-xgrid[Xind] + X)[:, :, np.newaxis]
    y2 = (ygrid[Yind + 1] - Y)[:, :, np.newaxis]
    y1 = (-ygrid[Yind] + Y)[:, :, np.newaxis]

    N = dx[Xind] * dy[Yind]

    V = (x2 * y2 * f11 + x1 * y2 * f12 + x2 * y1 * f21 + x1 * y1 * f22) / N[
        :, :, np.newaxis
    ]

    if extrapolation_value is not None:
        # if we need to fill the extrpolated values, find their indices here
        extrapolated_indices = (
            (X < np.min(xgrid))
            | (Y < np.min(ygrid))
            | (Y > np.max(ygrid))
            | (X > np.max(xgrid))
        )
        V[extrapolated_indices, ...] = extrapolation_value

    return V


# %%
def extract_submatrix(M, row_indices, col_indices, n2, nr, nc):
    """Extracts the submatrix corresponding to the row indices and column indices provided. The results
    is equivalent to `M[np.ix_(row_indices, col_indices)]`, but appears to be about 25% faster by flattening
    `M` and using the linear indices, then reshaping.
    See also discussion here: https://stackoverflow.com/a/14387955

    Parameters
    ----------
    M : np.array
        The matrix from which to extract the submatrices
    row_indices, col_indices : list
        row and column indices to be included in the submatrix
    n2, nr, nc : int
        n2 = M.shape[1], nr = len(row_indices), nc = len(col_indices)

    Returns
    -------
    np.array
        shape (nr, nc) array extracted from `M`
    """
    # not computing this saves some time:
    # n1, n2 = M.shape
    # nq = row_indices.size
    # nk  = col_indices.size
    # if we could rely on getting np.arrays instead of lists, this would also be unnecesarry:
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)
    return M.ravel()[(col_indices + n2 * row_indices.reshape((-1, 1))).ravel()].reshape(
        (nr, nc)
    )


# %%
def unpack_cv2_SIFT_octave(packed_octave):
    """cv2.SIFT writes the "packed octave" (see [issue here](https://github.com/opencv/opencv/issues/4554)) into the
    cv2.KeyPoint's `octave` attribute. If you want to use the SIFT-detected keypoints with a different descriptor (e.g. ORB),
    you'll have to unpack first, else you get weird '(-215:Assertion failed) inv_scale_x > 0 in function 'resize'' errors.
    NOTE: Likely you have to "pack" the octave if you want to use another detector with SIFT descriptors..."""
    octave = packed_octave & 255
    if octave >= 128:
        octave |= -128
    # Above means: ocvtave 128 becomes -128, 129 becomes -127, ..., 255 becomes -1
    # has to do with "two's complement" notation, but I can't say I fully get that.
    layer = (packed_octave >> 8) & 255
    scale = 1.0 / (1 << octave) if octave >= 0 else 1.0 * (1 << -octave)
    return (octave, layer, scale)


if __name__ == "__main__":
    # Make an image with a red, blue, green, white corner, and interpolate
    V0 = (
        np.dstack(
            (
                np.array([[1, 0], [0, 1]]),
                np.array([[0, 1], [0, 1]]),
                np.array([[0, 0], [1, 1]]),
            )
        )
        * 255
    )
    X0, Y0 = (0, 1), (0, 1)
    X, Y = np.meshgrid(np.linspace(0, 1, num=25), np.linspace(0, 1.1, num=30))
    V = bilin_interp(Y, X, Y0, X0, V0)
    # look at it with plt.imshow(V.astype(np.uint8))
    # Note that the funny bit for Y > 30/1.1 is expected, since the data there is exrapolated
