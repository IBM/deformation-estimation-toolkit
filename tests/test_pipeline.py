import os

import cv2
import numpy as np
import pytest

from .context import extract_class_name

import endeform.reflections.glare_detection as glares

GLARES = (glares.green_glare_mask,)

import endeform.detectors_descriptors.detect as ddd
try:
    from endeform.detectors_descriptors import xdetect
    XDETECT_AVAILABLE = True
except ImportError:
    xdetect = None
    XDETECT_AVAILABLE = False

DETECTORS = (ddd.cv2AKAZE(threshold=0.0001),) + \
    ( (xdetect.PatchORB(),xdetect.PyramidORB() ) if XDETECT_AVAILABLE else ()
    )
# AKAZE with default parameters only around 10 keypoints
#  , ddd.cv2SIFT() -> SIFT does funny stuff with octave ("packs" it), and also has negative octaves (which is fine, but ORB can't deal with it)
DESCRIPTORS = (ddd.cv2LATCH(), ) + \
    ( (xdetect.PatchORB(),xdetect.PyramidORB() ) if XDETECT_AVAILABLE else ()
    )


#  , ddd.cv2SIFT() -> SIFT probably expects a packed octave. Not worth the hassle, so let's maybe not use SIFT.

import endeform.interpolation.TPS as TPS
import endeform.interpolation.rigid as rigid

INTERPOLATORS = (
    TPS.TPS(),
    rigid.HomographyInterpolator(),
    rigid.TranslationInterpolator(),
    rigid.ProcrustesInterpolator(),
    rigid.AffineInterpolator(),
)

import endeform.filters.match_filters as mfilt

MFILTERS = (
    mfilt.best_N(),
    mfilt.absolute_translation_filter(),
    mfilt.average_translation_filter(),
)

import endeform.matchers.match as match

# Matchers have a binary and a real version. Group the 2 versions (Regular, Binary)
MATCHERS = (
    (match.BruteForceMatcher(), match.BruteForceMatcherBinary()),
    (match.LoweRatioMatcher(), match.LoweRatioMatcherBinary()),
)

from endeform.pipeline import Pipeline

from .context import SAMPLE_DATA_FOLDER

img1 = cv2.imread(os.path.join(SAMPLE_DATA_FOLDER, "Pat41_frame1_slight_deform.PNG"))
img2 = cv2.imread(os.path.join(SAMPLE_DATA_FOLDER, "Pat41_frame2_slight_deform.PNG"))
h, w, *_ = img1.shape
# Just pick a few points to eval on, we don't have that much time ;-)
Ny, Nx = 4, 5
yy, xx = np.meshgrid(np.arange(h, step=h // Ny), np.arange(w, step=w // Nx))
XY = np.column_stack((xx.flat, yy.flat))

@pytest.mark.parametrize("glare", GLARES)
@pytest.mark.parametrize("detector", DETECTORS, ids=extract_class_name)
@pytest.mark.parametrize("matcher", MATCHERS, ids=extract_class_name)
@pytest.mark.parametrize("mfilter", MFILTERS, ids=extract_class_name)
@pytest.mark.parametrize("interpolator", INTERPOLATORS, ids=extract_class_name)
@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=extract_class_name)
def test_elements_in_Pipeline(
    glare, detector, descriptor, matcher, mfilter, interpolator
):
    """Systematically tests all combinations of available elements in a pipeline"""
    # make sure to use the right matcher:
    if descriptor.isbinarydescriptor():
        matcher = matcher[1]
    else:
        matcher = matcher[0]

    P = Pipeline(
        keypoint_mask=glare,
        detector=detector,
        descriptor=descriptor,
        matcher=matcher,
        match_filter=mfilter,
        interpolator=interpolator,
    )

    # initialize
    P.init(img1)
    # take one step
    P.step(img2)
    # eval
    Q = P.eval(XY)

    # assert that attributes have been updated
    for attr in (
        P._last_keypoints,
        P._last_descriptors,
        P.initial_keypoints,
        P.initial_keypoint_descriptors,
        P._last_matches,
        P._last_image,
        P.initial_image,
    ):
        assert attr is not None

    np.testing.assert_array_equal(Q.shape, XY.shape)

    # Reasons for failure:
    # 1. Homography isn't found. That seems to happen here:
    #   - if there are less than 8 matches. E.g. AKAZE with default detects very few keypoints, so that this situation can occur
    # 2. Incompatibilities between different detectors/descriptors from OpenCV
    #   - SIFT "packs" the octave, so that keypoint.octave has gigantic values, which other descriptors don't understand. See helpers.utils.unpack_cv2_SIFT_octave.
    #       I assume that also means SIFT can only be used with its own detector, so I removed SIFT from the tests.
    #   - ORB doesn't accept negative octave (corresponds to upscaled image), even though that is perfectly fine afaik
    #   - I'm sure there's more....
