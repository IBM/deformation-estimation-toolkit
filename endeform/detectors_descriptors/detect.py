"""
Object detection & description functions and classes
"""
import cv2
import numpy as np
    
from . import DD_BaseClasses as Dbd
from ..helpers.image_helpers import tiler as _tiler


# %% Custom detectors/descriptors

# %%  Wrappers around openCV detectors/descriptors
class cv2LATCH(Dbd.BaseBinaryDescriptor):
    """A wrapper around cv2.xfeatures2d_LATCH. Please see the cv2 help for details. As far as I can tell, Descriptor ignores keypoint size.

    Attributes
    ----------

    Parameters
    -------
    Note: this is reproduced from the `OpenCV docs <https://docs.opencv.org/4.4.0/d6/d36/classcv_1_1xfeatures2d_1_1LATCH.html>`
    bytes : 64, 32, 16, 8, 4, 2 or 1
        is the size of the descriptor
    rotationInvariance :
        whether or not the descriptor should compensate for orientation changes
    half_ssd_size :
        the size of half of the mini-patches size. For example, if we would like to compare triplets of patches of size 7x7x then the half_ssd_size should be (7-1)/2 = 3
    sigma :
        sigma value for GaussianBlur smoothing of the source image. Source image will be used without smoothing in case sigma value is 0.

    Examples
    --------
    """

    def __init__(self, *args, **kwargs):
        self.descriptor = cv2.xfeatures2d.LATCH_create(*args, **kwargs)

    def compute(self, img, keypoints, **kwargs):
        """Computes the descriptors of `keypoints`

        Parameters
        ----------
        img : np.array
            The image, either BGR or grey
        keypoints : list of cv2.KeyPoint
            The keypoint locations

        Returns
        -------
        list of cv2.KeyPoint
            The subset of `keypoints` for which descriptors were computed
        np.array
            The descriptors.
        """
        kpts, descs = self.descriptor.compute(
            img,
            keypoints,
        )
        return kpts, descs


class cv2SIFT(Dbd.BaseDetectorDescriptor):
    """
    cv2SIFT detector
    """

    def __init__(self, *args, **kwargs):
        self.detectordescriptor = cv2.SIFT_create(*args, **kwargs)

    def detect_and_compute(self, img, mask=None, *args, **kwargs):
        """
        cv2SIFT detect_and_compute
        """
        return self.detectordescriptor.detectAndCompute(
            img, mask if mask is None else mask.astype(np.uint8), *args, **kwargs
        )

    detectAndCompute = detect_and_compute

    def detect(self, img, mask=None, *args, **kwargs):
        """
        cv2SIFT detect
        """
        return self.detectordescriptor.detect(
            img, mask if mask is None else mask.astype(np.uint8), **kwargs
        )

    def compute(self, img, keypoints, **kwargs):
        """
        cv2SIFT compute
        """
        return self.detectordescriptor.compute(
            img,
            keypoints,
        )


# %%
class cv2AKAZE(Dbd.BaseDetectorBinaryDescriptor):
    def __init__(self, *args, **kwargs):
        self.detectordescriptor = cv2.AKAZE_create(
            *args, descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, **kwargs
        )
        # cv2.AKAZE_DESCRIPTOR_KAZE is not a binary descriptor

    def detect_and_compute(self, img, mask=None, *args, **kwargs):
        return self.detectordescriptor.detectAndCompute(
            img, mask if mask is None else mask.astype(np.uint8), *args, **kwargs
        )

    detectAndCompute = detect_and_compute

    def detect(self, img, mask=None, *args, **kwargs):
        return self.detectordescriptor.detect(
            img, mask if mask is None else mask.astype(np.uint8), **kwargs
        )

    def compute(self, img, keypoints, **kwargs):
        return self.detectordescriptor.compute(
            img,
            keypoints,
        )
