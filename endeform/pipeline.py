import numpy as np

import endeform.helpers.plot_helpers as ph


#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
def _test_iterable(X):
    # Returns True if `iter(X)` doesn't throw an error, else False
    try:
        iter(X)
        return True
    except TypeError:
        return False


#%%
class Pipeline:
    """Implements the complete pipeline to estimate a deformation between frame 0
    and some next frame.
    Mandatory components are a detector, descriptor, matcher, and interpolator, the
    other components are optional.

    Steps:
    image -> image_filter(s) -> keypoint_mask(s) -> image_filter_after_mask
        -> detector -> descriptor (or detectordescriptor, see below)
        -> keypoint_filter -> matcher -> match_filter(s) -> interpolator

    NOTE: if `detector` and `descriptor` are the same object (and an instance of
    detect.DD_BaseClasses.BaseDetectorDescriptor), then instead of calling `detect()`
    and then `compute()`, `detect_and_compute()` is called.
    NOTE: for components where the plural is indicated, iterables can be provided.
    Their elements will be called one after the other. E.g. it is possible to specify
    filtering matches in two steps, by first keeping only the best 100 matches and
    then removing the outliers in terms of translation.

    Parameters
    ----------
        detector : detect.BaseDetector
            The detector to be used. Needs to have at least a `detect(image, mask)` method which returns a list of `cv2.KeyPoint`
        descriptor : detect.BaseDescriptor
            The descriptor used. Needs to have at least a `compute(image, list of cv2.KeyPoint)` method, which computes the descriptors
            of the keypoints in the image supplied
        NOTE: if `detector is descriptor`, i.e. they're the same object, and they're a subclass of `BaseDetectorDescriptor`,
            then `detect_and_compute()` is called once instead of `detect()` followed by `compute()`.
        matcher : matchers.BaseMatcher
            The keypoint matcher. Needs to have at least a `match(descriptors1, descriptors2)` method, which finds
            the keypoints in the 1st array that match the keypoints in the second array and returns a list of cv2.DMatch
        interpolator : BaseInterpolator
            The interpolator. Needs to have at least a `fit(coordinates, values)` methods, which will fit the internal parameters so
            that `eval(query_coordinates)` will yield values which are interpolated from the tuples `coordinates[i,:], values[i,:]`.
        image_filter : Callable or Iterable or None, optional
            A function or Iterable of functions with the signature func(img: np.array) -> np.array of the same shape, i.e. it applies a filter
            to the supplied image, e.g. a denoising operation, by default None
        keypoint_mask : Callable or None, optional
            A function with the signature func(img: np.array with shape (h,w,c)) -> np.array of Booleans with shape (h,w). The output
            is used as a mask for each frame, wherever it is True, no keypoints are detected. By default None
        image_filter_after_mask : Callable or None, optional
            Same as `image_filter`, only this one is applied after the keypoint mask is computed, if any; by default None
        keypoint_filter : Callable or None , optional
           A function with the signature func(keypoints: list of cv2.KeyPoint, descriptors: np.array) -> (list of cv2.KeyPoint, np.array).
           Filters keypoints, e.g. by removing almost-collocated keypoints, or the ones with the smallest "cornerness". The output of this
           function, if given, is used for matching in place of the supplied keypoints and descriptors. By default None
        match_filter : object or Iterable, optional
            Each match filter needs to have a `filter()` method with the signature
            filter(matches: list of cv2.DMatch, keypoints1: np.array, keypoints2: np.array) -> list of cv2.DMatch.
            The returned list of matches will be used instead of the suplied matches. By default None
    Attributes
    ----------
        _detector :
            The object used for keypoint detection, or None
        _descriptor :
            The object used for keypoint description, or None
        _detectordescriptor :
            The detector/descriptor used, if `detector is descriptor`, see explanation in Parameters section; or None.
            We always have `(_detectordescriptor is None and _detector is not None and _descriptor is not None) or
            (_detectordescriptor is not None and _detector is None and _descriptor is None)`
        _matcher :
            The object used to match the keypoints found in pairs of frames (typically intial frame
            and frame `t`)
        _interpolator :
            The object performing the interpolation between the translations of the matched keypoints.
        _image_filter, _image_filters :
            Filter applied to the image _before_ mask computation
            If there's only a single image filter specified, it is _image_filter,
            and _image_filters in False.
            If there are multiple filters, they are in the (iterable) _image_filters,
            and _image_filter is False.
        _keypoint_mask :
            Computes a mask in which no keypoint will be detected. Typically will detect reflections, instruments, etc
        _image_filter_after_mask :
            Filters applied to the image _after_ mask computation
        _keypoint_filter :
            Filter keypoints, e.g. by removing almost-collocated keypoints, or the ones with the smallest "cornerness"
        _match_filter, _match_filters :
            Filter(s) applied to matches, e.g. by removing matches corresponding to excessive deformation (and hence
            likely erroneous)
            see `_image_filter(s)` above for explanation the plural form.
        initial_image :
            The initial image. All following images are deformed onto this one
        initial_keypoints, initial_descriptors :
            The keypoints and descriptors computed in the initial image
        _last_image, _last_keypoints, _last_descriptors, _last_matches :
            The last image processed (i.e. whose deformation with respect to the
            intial image was estimated), along with the keypoints, descriptors, and
            matches used.
    Methods
    -------
        init(initial_image)
            Initialize the pipeline once the initial image is available
        step(next_image)
            Estimate the deformation between initial_image and next_image
        eval(query_points: np.array)
            Evaluate the deformation at the query_points, i.e. for every point
            x,y = query_points[i, :]
            compute the estimated new location it was deformed to, i.e.
            x_new, y_new = deformation(x,y)
        draw_matches()
            draw the matches used for the last deformation estimation
    Examples
    --------
    """

    def __init__(
        self,
        *,
        detector,
        descriptor,
        matcher,
        interpolator,
        image_filter=None,
        keypoint_mask=None,
        image_filter_after_mask=None,
        keypoint_filter=None,
        match_filter=None,
    ):
        # TODO: allow Iterables for:
        # keypoint_mask (needs merging strategy),
        # keypoint_filter
        self._matcher = matcher
        self._interpolator = interpolator
        if detector is descriptor:
            self._detectordescriptor = detector
            self._detector = None
            self._descriptor = None
            self._detect_and_describe = self.__detect_and_describe
        else:
            self._detectordescriptor = None
            self._descriptor = descriptor
            self._detector = detector
            self._detect_and_describe = self.__detect_then_describe
        # populate the filters and masks.
        # Plural attributes if they're iterables, singular set to False
        # otherwise singular attributes, and plural set to False
        if _test_iterable(image_filter):
            self._image_filter = False
            self._image_filters = image_filter
        else:
            self._image_filter = image_filter
            self._image_filters = False
        if _test_iterable(match_filter):
            self._match_filter = False
            self._match_filters = match_filter
        else:
            self._match_filter = match_filter
            self._match_filters = False

        self._image_filter_after_mask = image_filter_after_mask
        self._keypoint_mask = keypoint_mask
        self._keypoint_filter = keypoint_filter

    def init(self, frame0):
        """Perform necessary initialization of the pipeline: Store `initial_image`, compute `initial_keypoints`
        and `initial_descriptors` and store them

        Parameters
        ----------
        frame0 : np.array
            The initial frame, shape is either (h,w) or (h,w,3).

        Returns
        -------
        self
        """
        # init
        self.initial_image = frame0.copy()
        kpts0, desc0 = self._process_single_image(self.initial_image)
        self.initial_keypoints = kpts0
        self.__initial_keypoints_as_np_array = np.array([kpt.pt for kpt in kpts0])
        self.initial_keypoint_descriptors = desc0
        return self

    def _process_single_image(self, img):
        """Process an image up to the point where the keypoints are available"""
        # first image filter
        if self._image_filter is not None:
            if self._image_filter:
                img = self._image_filter(img)
            else:
                for filter in self._image_filters:
                    img = filter(img)
        # generate mask for keypoints, e.g. glare filter
        if self._keypoint_mask is not None:
            self._last_mask = ~(self._keypoint_mask(img))
            # Don't forget that cv2 wants the mask to be True wherever
            # keypoints are being looked for
        else:
            self._last_mask = None
        if self._image_filter_after_mask is not None:
            img = self._image_filter_after_mask(img)
        kpts, desc = self._detect_and_describe(img=img, mask=self._last_mask)
        if self._keypoint_filter is not None:
            kpts, desc = self._keypoint_filter(kpts, desc, mask=self._last_mask)
        self._last_keypoints = kpts
        self._last_descriptors = desc
        return kpts, desc

    def _match_two_images_and_compute_interpolator(self, kpts, desc):
        """With two images and their keypoints available, now we
        perform matching and interpolation.
        The first image is always the initial image"""
        matches = self._matcher.match(self.initial_keypoint_descriptors, desc)
        if self._match_filter is not None:
            if self._match_filter:
                matches = self._match_filter.filter(
                    matches, self.initial_keypoints, kpts
                )
            else:
                for filter in self._match_filters:
                    matches = filter.filter(matches, self.initial_keypoints, kpts)
        self._last_matches = matches
        self._interpolator.fit(
            self.__initial_keypoints_as_np_array[[m.queryIdx for m in matches], :],
            np.array([kpts[m.trainIdx].pt for m in matches]),
        )

    def step(self, img):
        """Compute the keypoints of `img`, match them with the ones from `self.initial_image` and fit the interpolator.
        Also updates all the attributes starting with `_last...`. After `step(img)` has been called, you can use `eval()`.

        Parameters
        ----------
        img : np.array
            The next frame

        Returns
        -------
        self
        """
        self._last_image = img
        kpts, desc = self._process_single_image(img)
        self._match_two_images_and_compute_interpolator(kpts, desc)
        return self

    def eval(self, query_points):
        """Evaluate the estimated deformation between `self.initial_image` and `self._last_image` based on the motion of the
        detected matches between keypoints in both images. Requires you to call `fit()` first!

        Parameters
        ----------
        query_points : np.array
            Shape (N,2) array, each row `query_points[i,:]` corresponds to a point at which the estimated deformation is evaluated,
            i.e. the point `query_points[i,:]` has moved to `eval(query_points)[i,:]`

        Returns
        -------
        np.array
            Shape (N,2) array of the new locations of `query_points` after the estimated deformation is applied.
        """
        Z = self._interpolator.eval(query_points)
        return Z

    def __detect_then_describe(self, img, mask=None):
        kpts1 = self._detector.detect(img=img, mask=mask)
        kpts1, desc1 = self._descriptor.compute(img=img, keypoints=kpts1)
        return kpts1, desc1

    def __detect_and_describe(self, img, mask=None):
        kpts1, desc1 = self._detectordescriptor.detect_and_compute(img=img, mask=mask)
        return kpts1, desc1

    #%% some convenience functions
    def draw_matches(self, **kwargs):
        """calls `endeform.plot_helpers.draw_matches` using the last
        processed image pair. See the `draw_matches` docstring for keyword args
        """
        ph.draw_matches(
            self.initial_image,
            self._last_image,
            self.initial_keypoints,
            self._last_keypoints,
            self._last_matches,
            **kwargs,
        )


class PipelinePrecompute(Pipeline):
    """Pipeline object with interpolator that implements a precomputed RBF matrix.
    For now the only one available is TPS.TPSPrecompute. See Pipeline for the docstring

    See Also:
    ---------
    Pipeline : Parent class, which doesn't implement any precomputation."""

    def __init__(
        self,
        *,
        detector,
        descriptor,
        matcher,
        interpolator,
        query_points,
        image_filter=None,
        keypoint_mask=None,
        image_filter_after_mask=None,
        keypoint_filter=None,
        match_filter=None,
    ):
        super().__init__(
            detector=detector,
            descriptor=descriptor,
            matcher=matcher,
            interpolator=interpolator,
            image_filter=image_filter,
            keypoint_mask=keypoint_mask,
            image_filter_after_mask=image_filter_after_mask,
            keypoint_filter=keypoint_filter,
            match_filter=match_filter,
        )
        self.query_points = query_points

    def init(self, frame0):
        """Initializes the Pipeline (see docstring there) and additionally precomputes the RBF (Radial Basis Function)
        matrix on `self.query_points`. This can take a while, but makes every `eval()` and `step()` call faster.

        Parameters
        ----------
        frame0 : np.array
            The initial image, shape is either (h,w) or (h,w,3).

        Returns
        -------
        self
        """
        super().init(frame0)
        # every "private" attribute will be "mangled", and it makes sense
        # to expose this attribute anyway:
        self._precomputed_knots = self._Pipeline__initial_keypoints_as_np_array
        self._interpolator.precompute_RBF(
            pre_query_points=self.query_points, pre_knots=self._precomputed_knots
        )
        return self

    def _match_two_images_and_compute_interpolator(self, kpts, desc):
        """With two images and their keypoints available, now we
        perform matching and interpolation.
        The first image is always the initial image"""
        matches = self._matcher.match(self.initial_keypoint_descriptors, desc)
        if self._match_filter is not None:
            if self._match_filter:
                matches = self._match_filter.filter(
                    matches, self.initial_keypoints, kpts
                )
            else:
                for filter in self._match_filters:
                    matches = filter.filter(matches, self.initial_keypoints, kpts)
        self._last_matches = matches
        self._last_matched_queryIdxs = tuple(m.queryIdx for m in matches)
        self._interpolator.fit(
            knot_indices=self._last_matched_queryIdxs,
            y=np.array([kpts[m.trainIdx].pt for m in matches]),
        )

    def eval(self, query_indices=None):
        """Evaluate the resulting deformation on the pre-computed query points `self.query_points`
        (or the subset indicated by `query_indices`)

        Parameters
        ----------
        query_indices : list of int, optional
            The set of indices of the query points on which the RBF has been precomputed on
            which the interpolator should be evaluated, by default None, which the deformation
            of all points `self.query_points` is computed.

        Returns
        -------
        np.array
            len(query_indices) by 2 array, the deformed positions of the points
            self.query_points[query_indices,:].

        See Also:
        --------
        Pipeline : Parent class, slower but more flexible.
        """
        Z = self._interpolator.eval_pre(query_indices=query_indices)
        return Z
