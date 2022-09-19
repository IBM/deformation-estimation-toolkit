"""Interpolators which fit one of several rigid transformations. In increasing generality:
- TranslationInterpolator fits a translation only, does not change any shapes, nor their orientation
- ProcrustesInterpolator fits a Procrustes transform, i.e. a combination of translation, rotation, and UNIFORM scaling, i.e. shapes do not change save for scaling, angles do not change
- AffineInterpolator fits an affine transform, which allows anisotropic scaling, and keeps parallel lines parallel
- HomographyInterpolator fits a homography, which will keep straight line straight, but does not preserve parallelism etc"""

import cv2

#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from .I_BaseClasses import BaseInterpolator


# %%
class BaseRigidInterpolator(BaseInterpolator):
    def __init__(self) -> None:
        super().__init__()
        self.coef_ = None

    def eval(self, query_points):
        """Evaluate the rigid transfrom on the `query_points`.

        Parameters
        ----------
        query_points : np.array
            Array of shape (N, num_dims). The array of points on which the transformation is evaluated

        Returns
        -------
        np.array
            Array of shape (N, num_dims). The new locations of `query_points` after being transformed.

        Notes
        -----
        Except for `HomographyInterpolator`, `self.coef_` is a Matrix [ A.T | t.T], where the last row t is the translation, and the mapping is
        `new_location = old_location @ A.T + t.T` (`query_points.shape = (N, num_dims)`, and `t.T` is broadcasted).
        For `HomographyInterpolator`, the mapping is between homogenous coordinates
        `new_location_unscaled = [old_location | 1] @ self.coef_ ` and
        `new_location = new_location_unscaled[:,:2] / new_location_unscaled[:,[2]]` (note the broadcasting, again.)
        """

        return query_points @ self.coef_[:-1, :] + self.coef_[[-1], :]

    def fit(self, X_from: np.array, X_to: np.array, **kwargs):
        """Find the transform which moves `X_from` into `X_to`.

        Parameters
        ----------
        X_from, X_to : np.array
            Arrays of equal shape (N, num_dims)

        Returns
        -------
        Instance
            self
        """
        raise NotImplementedError


class TranslationInterpolator(BaseRigidInterpolator):
    """Implements an "interpolator" that only finds the "best" translation, either the (coordinate-wise) median, or the mean translation.

    Since there is an abundance of robust averages, including several definitions of the median for n>1, truncated and censored means, outlier detection, ...
    this Interpolator could get a lot more functionality in the future."""

    _available_norms = (1, 2)

    def __init__(self, norm=1) -> None:
        """Interpolator finding an average translation

        Parameters
        ----------
        norm : int, optional
            1 for median, 2 for mean translation, by default 1

        Raises
        ------
        ValueError
            if `norm not it {1,2}`
        """
        super().__init__()
        if not norm in self._available_norms:
            raise ValueError(
                f"{norm} for argument `norm` is not valid. Currently available is {self._available_norms}"
            )
        else:
            self.norm = norm

    def fit(self, X_from, X_to):
        n_ = X_from.shape[1]  #  number of dimensions
        try:
            np.testing.assert_equal(X_from.shape, X_to.shape)
        except AssertionError as e:
            raise ValueError(
                f"{X_from.shape} and {X_to.shape} are not equal, but `X_from` and `X_to` must have equal shapes."
            )

        if self.norm == 1:
            t = np.median(X_to - X_from, axis=0, keepdims=True)
        elif self.norm == 2:
            t = np.mean(X_to - X_from, axis=0, keepdims=True)
        else:
            raise NotImplementedError

        self.coef_ = np.vstack((np.eye(n_), t))


# %%
class __cv2_BasedRigidInterpolator(BaseRigidInterpolator):
    """Base class for an estimator wrapped around the rigid transformations implemented in OpenCV [1]_

    Attributes
    ----------
    _inliers : list or None
        The robust methods (RANSAC and LMedS) also identify inliers. After `fit(X_from, X_to)` was performed, this `_inliers` holds the indices corresponding to inliers, i.e. the
        row indices in `X_from` and `X_to` which where actually used to compute the transformation.
    _coef : the 3-by-3 matrix parametrizing the transformations

    Methods
    -------
    fit : compute the transformation parameters
    eval : after `fit` was called, you can evaluate the transformation

    Examples
    --------
    """

    def __init__(self, *, method="ransac", **kwargs) -> None:
        """

        Parameters
        ----------
        method : str, optional
            Can be RANSAC or LMEDS, and for the Homography Estimator "0" (just simple least-squares) and "RHO"  (PROSAC-based method) are also available, by default 'ransac'

        .. [1] OpenCV documentation https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
        """
        super().__init__()
        self.method = method.lower()
        if self.method == "ransac":
            self.__method = cv2.RANSAC
        elif self.method == "lmeds":
            self.__method = cv2.LMEDS
        elif self.method == "0":
            self.__method = 0
        elif self.method == "rho":
            self.__method = cv2.RHO
        else:
            raise NotImplementedError(
                f'{method} is not a known method. Only "ransac", "lmeds", "0" and "rho" are valid.'
            )
        self.__kwargs = kwargs

    def _cv2_estimator(self):
        raise NotImplementedError

    def fit(self, X_from, X_to):
        try:
            np.testing.assert_equal(X_from.shape, X_to.shape)
        except AssertionError as e:
            raise ValueError(
                f"{X_from.shape} and {X_to.shape} are not equal, but `X_from` and `X_to` must have equal shapes."
            )
        transformation, inliers = self._cv2_estimator(
            X_from, X_to, method=self.__method, **self.__kwargs
        )
        self.coef_ = transformation.T
        self._inliers = inliers
        return self


# %%
class ProcrustesInterpolator(__cv2_BasedRigidInterpolator):
    """Estimates a Procrustes transformation, i.e. a transformation combinining _uniform_ scaling, rotation, and translation.
    This is mostly a wrapper around cv2.estimateAffinePartial2D
    TODO: Docstring from __cv2_BasedRigidInterpolator applies here, too"""

    def _cv2_estimator(self, *args, **kwargs):
        return cv2.estimateAffinePartial2D(*args, **kwargs)


class AffineInterpolator(__cv2_BasedRigidInterpolator):
    """Estimates an affine transformation.
    This is mostly a wrapper around cv2.estimateAffine
    TODO: Docstring from __cv2_BasedRigidInterpolator applies here, too"""

    def _cv2_estimator(self, *args, **kwargs):
        return cv2.estimateAffine2D(*args, **kwargs)


# %%
class HomographyInterpolator(__cv2_BasedRigidInterpolator):
    """Estimates a Homography (aka Perspective Transformation), which has 8 degrees of freedom and is represented by
    a 3x3 matrix with (3,3) element normalized to 1.
    TODO: Docstring from __cv2_BasedRigidInterpolator applies here, too"""

    def _cv2_estimator(self, *args, **kwargs):
        return cv2.findHomography(*args, **kwargs)

    def fit(self, X_from, X_to):
        super().fit(X_from, X_to)
        # for homography, the interpretation of coef_ changes, it isn't "[A | t]" anymore, so no need to transpose.
        self.coef_ = self.coef_.T
        return self

    # eval needs to be overridden
    def eval(self, query_points):
        return cv2.perspectiveTransform(
            query_points[np.newaxis, ...].astype(float), self.coef_
        ).squeeze()
        # ^ np.newaxis and squeeze() necessary, because cv2.perspective transform expects (1, num_datapoints, num_dimensions) array
        # and also the explicit casting to float, if we still happen to have ints, OpenCV will throw an error...
