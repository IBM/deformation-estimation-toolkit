from itertools import compress
from warnings import warn

import cv2

from . import DM_BaseClasses as Dmb


class BruteForceMatcher(Dmb.BaseMatcher):
    """A wrapper for cv2.BFMatcher. Note that the default normType is cv2.NORM_L2, appropriate
    for e.g. SIFT, but not for binary descriptors.
    """

    def __init__(self, *args, **kwargs):
        # overrides the cv2 default: if crossCheck is not specfied, set it to True:
        kwargs["crossCheck"] = kwargs.get("crossCheck", True)
        self.matcher = cv2.BFMatcher_create(*args, **kwargs)

    def match(self, descriptors1, descriptors2, mask=None):
        """
        Parameters
        ----------
        descriptors1, descriptors2 : np.array
            Each row corresponds to a keypoint's descriptor. In cv2 terminology, descriptors1 are
            "query" descriptors, descriptors2 "train" descriptors
        mask : [type], None or np.uint8
            If mask[i,j]==0, descriptors1[i,:] cannot be matched to descriptors2[j,:],
            by default None
        Returns
        -------
        List of cv2.DMatch
            Each element a match. descriptors1[match.queryIdx,:] is matched with
            descriptors2[match.trainIdx,:].
            match.distance is the distance between the matches (in descriptor space!)
            Note: OpenCV 4.5.4 replaced most lists with tuples, this method casts to
            list explicitly. 
        """

        # Could check here whether normType and descriptors.dtype are
        # compatible
        return list(self.matcher.match(descriptors1, descriptors2, mask))


# Subclass for binary descriptors
class BruteForceMatcherBinary(BruteForceMatcher):
    """A wrapper for cv2.BFMatcher using normType=cv2.NORM_HAMMING, appropriate for binary
    descriptors
    """

    def __init__self(self, *args, **kwargs):
        if kwargs.get("normType", cv2.NORM_HAMMING) is not cv2.NORM_HAMMING:
            raise RuntimeError(
                """Do not set normType manually when using BruteForceMatcherBinary, 
                            it is automatically set to cv2.NORM_HAMMING"""
            )
        kwargs["normType"] = cv2.NORM_HAMMING
        super().__init__(*args, **kwargs)


#
class LoweRatioMatcher(Dmb.BaseMatcher):
    """
    Implements the Lowe Ratio Matcher, i.e. match is discarded if second-best match is almost as good as best.
    """

    #NOTE: `ratio` needs to be treated differently from the remainin kwargs:
    #   kwargs are passed on to the BruteForceMatcher() call
    def __init__(self, *args, ratio=.75, **kwargs):
        # super().__init__()
        if kwargs.get("crossCheck", False):
            warn(
                "'crossCheck' has to be False for k-NN matching with k>1. Setting it to False."
            )
        kwargs["crossCheck"] = False
        self.matcher = cv2.BFMatcher_create(*args, **kwargs)
        self.ratio = ratio

    def match(self, descriptors1, descriptors2, ratio=None, mask=None):
        """Match descriptors by, for each descriptor in `descriptors1`, first finding the 2 closest
        (in descriptor space) descriptors in `descriptors2` for it, and then only accepting the match if
            (distance to closest match) < (distance to 2nd closest match) * ratio
        else, the match is considered too ambiguous.

        Parameters
        ----------
        descriptors1, descriptors2 : np.array
            Each row corresponds to a keypoint's descriptor. In cv2 terminology, descriptors1 are
            "query" descriptors, descriptors2 "train" descriptors
        ratio : float
            Positive, less than 1. Default: None, i.e. use self.ratio
        mask : [type], None or np.uint8
            If mask[i,j]==0, descriptors1[i,:] cannot be matched to descriptors2[j,:],
            by default None
        Returns
        -------
        List of cv2.DMatch
            Each element a match. descriptors1[match.queryIdx,:] is matched with
            descriptors2[match.trainIdx,:].
            match.distance is the distance between the matches (in descriptor space!)
        """
        knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        if ratio is None:
            ratio = self.ratio
        boolean_index = map(
            lambda best2matches: best2matches[0].distance
            < ratio * best2matches[1].distance,
            knn_matches,
        )
        return [matches[0] for matches in compress(knn_matches, boolean_index)]


class LoweRatioMatcherBinary(LoweRatioMatcher):
    """A wrapper for `LoweRatioMatcher` using normType=cv2.NORM_HAMMING, appropriate for binary
    descriptors
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("normType", cv2.NORM_HAMMING) is not cv2.NORM_HAMMING:
            warn(
                """Setting normType to cv2.NORM_HAMMING for BruteForceMatcherBinary 
                            """
            )
        """
        Use HAMMING normtype for binary descriptor matching
        """
        kwargs["normType"] = cv2.NORM_HAMMING
        super().__init__(*args, **kwargs)


class FLANNMatcher(Dmb.BaseMatcher):
    def __init__(self):
        raise NotImplementedError

    def match(self):
        pass
