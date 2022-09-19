"""Classes that can be used as the `match_filter` in `Pipelines`. Generally, they need to expose a `filter` method with the
    signature filter(matches: list of cv2.DMatch, keypoints1: np.array, keypoints2: np.array) -> list of cv2.DMatch

    Classes
    -------
    average_translation_filter :
        remove matches where the translation is considerably different from the average translation of all matches
    absolute_translation_filter :
        remove matches which exceed a spefied translation
    best_N :
        keep only the `N` matches with the smallest `distance` (in feature space, so e.g. for binary descriptors, that
        is the Hamming distance between the descriptors)
"""
#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
from itertools import compress  # to use boolean indexing with lists
from operator import attrgetter

import numpy as np


#%%
class average_translation_filter:
    """A class to filter matches whose x or y translation exceeds
        |translation| > `avg` + `spread`*`factor`

    Attributes
    ----------
    Methods
    -------
    filter(matches: List of cv2.Dmatch, kpts1, kpts2: list of keypoints among which the matches were made)
        Returns the list of filtered matches.
        NOTE: itertools.compress returns an iterator, I'm wrapping it in list() to return
                a list from the `filter` method, but maybe it could improve performance
                to work with the iterator instead. To be explored.
    Examples
    --------
    """

    __avgs = {"median": np.median, "mean": np.mean}
    __spreads = {
        "std": lambda x: np.std(x, axis=1, keepdims=True),
        "iqr": lambda x: np.diff(
            np.percentile(x, [25, 75], axis=1, keepdims=True), axis=0
        )[0],
    }
    # ^ Probably not the best way to provide flexibility in the choice of average and spread measures
    #  especially the acrobatics with axes and dimensions is awkward
    def __init__(self, avg="median", spread="IQR", factor=2):
        """[summary]

        Parameters
        ----------
        avg : str, optional
            'median' or 'mean', by default 'median'
        spread : str, optional
            'std' (Standard deviation) or 'IQR' (Interquartile Range), by default 'IQR'
        factor : float, optional
            factor to multiply `spread`; everything outside avg +/- factor*spread is filtered as an outlier
            , by default 2
        """
        self._avg = self.__avgs.get(avg.lower())
        self._spread = self.__spreads.get(spread.lower())
        self._factor = factor

    def filter(self, matches, kpts1, kpts2, inplace=False):
        """Filters the matches in `matches` by removing those who correspond to keypoint translations
        too different from the average translations. "Too different" measured as
        |translation| > self._avg(translations) + self._factor*self._spread(translations)

        Parameters
        ----------
        matches : list of cv2.DMatch
            The matches
        kpts1, kpts2 : list of cv2.KeyPoint
            The keypoints being matched, `kpts1` corresponds to match.queryIdx, `kpts2` to match.trainIdx
        inplace : bool, optional
            [Not implemented yet, just a placeholder] Change the input list instead of allocating
            and returning a new one, by default False

        Returns
        -------
        list of cv2.DMatch
            the list of filtered matches.
        """
        translations = np.array(
            [
                [kpts1[m.queryIdx].pt[i] - kpts2[m.trainIdx].pt[i] for m in matches]
                for i in (0, 1)
            ]
        )
        avgs = self._avg(translations, axis=1, keepdims=True)
        spreads = self._spread(translations)
        inliers_boolean = np.all(
            np.abs(translations - avgs) < self._factor * spreads, axis=0
        )
        return list(compress(matches, inliers_boolean))


#%%
class absolute_translation_filter:
    """Filters matches by their absolute deviation (in pixels) from the the average (median or mean)
    translation of all matches. NOTE: This assumes that the dominant motion is translation, it will
    perform poorly for rotations and extreme deformations.

    Attributes
    ----------
        d_max : 2-element np.array
            maximum deviation in x-, resp y-direction
    Methods
    -------

    Examples
    --------
    """

    # Really bad for rotations though!
    __avgs = {"median": np.median, "mean": np.mean}

    def __init__(self, d_max=(30, 30), avg="median"):
        """[summary]

        Parameters
        ----------
        d_max : 2-elem tuple or scalar, optional
            maximum deviation in pixels in x- and y-direction. A scalar is interpreted as same max deviation
            in both direction, by default (30,30)
        avg : str, optional
            The average to be used, available are 'mean' and 'median', by default 'median'
        """
        self.d_max = np.array(d_max).reshape(
            (-1, 1)
        )  #  will work for scalar d_max as well
        self._avg = self.__avgs.get(avg.lower())

    def filter(self, matches, kpts1, kpts2):
        """Filters the matches in `matches` by their absolute deviation (in pixels) from an average.
        The average is first computed as avg = self._avg( deviations_in_matches).
        A match is discarded if
            |translation[i] - avg[i]| > self.d_max[i]  for any i in {0,1}

        Parameters
        ----------
        matches : list of cv2.DMatch
            The input matches to be filtered
        kpts1, kpts2 : list of cv2.KeyPoint
            The keypoints being matched, `kpts1` corresponds to match.queryIdx, `kpts2` to match.trainIdx

        Returns
        -------
        list of cv2.DMatch
            The filtered matches
        """
        translations = np.array(
            [
                [kpts1[m.queryIdx].pt[i] - kpts2[m.trainIdx].pt[i] for m in matches]
                for i in (0, 1)
            ]
        )
        avgs = self._avg(translations, axis=1, keepdims=True)
        inliers_boolean = np.all(np.abs(translations - avgs) < self.d_max, axis=0)
        return list(compress(matches, inliers_boolean))


#%%
class best_N:
    """Filters matches by keeping only the best `N` (or less) matches

    Attributes
    ----------
        N : int
            number of matches to keeps
    Methods
    -------
        filter :
            Returns only the closest `N` matches.
    Examples
    --------
    """

    def __init__(self, N=50):
        """

        Parameters
        ----------
        N : int
            The best `N` matches will be kept. (by default: 50)
        """
        self.N = N

    def filter(self, matches, kpts1=None, kpts2=None):
        """Sorts `matches` by distance (lowest first) and returns only the
        closest `self.N` ones.
        NOTE: `matches` is sorted in-place.

        Parameters
        ----------
        matches : list of cv2.DMatch
            The input matches to be filtered
        kpts1, kpts2 : list of cv2.KeyPoint
            Ignored, just there for consistent API
        Returns
        -------
        list of cv2.DMatch
            The filtered matches
        """
        matches.sort(key=attrgetter("distance"))
        return matches[: self.N]
 #%%
#TODO:  class tukey_fence:
# Identifies outlier as anything outside [Q1 - k*IQR, Q3 + k*IQR], Q1, Q3 being the quartiles. Tukey suggested k=1.5 of "outlier". k=3 for "far out"
