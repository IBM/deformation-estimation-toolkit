import cv2
import numpy as np

import endeform.matchers.match as matchers

#%%
IDENTICAL_REAL_DESCRIPTORS = np.random.randint(0, high=220, size=(30, 128)).astype(
    np.float32
)
IDENTICAL_BINARY_DESCRIPTORS_UINT8 = np.random.randint(
    0, high=255, size=(40, 256), dtype=np.uint8
)
# recall that in openCV, binary descriptors are stored as uint8, so that [0,0,0,1,0,0,1,0] = 18 etc


def _template_test_identical_descriptors(matcher, descriptors):
    matches = matcher.match(descriptors, descriptors)
    np.testing.assert_array_equal(
        [m.queryIdx for m in matches], [m.trainIdx for m in matches]
    )


class TestBruteForceMatcher:
    def test_identical_descriptors_REAL(self):
        matcher = matchers.BruteForceMatcher(cv2.NORM_L2)
        _template_test_identical_descriptors(matcher, IDENTICAL_REAL_DESCRIPTORS)

    def test_identical_descriptors_BINARY(self):
        matcher = matchers.BruteForceMatcherBinary()
        _template_test_identical_descriptors(
            matcher, IDENTICAL_BINARY_DESCRIPTORS_UINT8
        )


class TestLoweRatioMatcher:
    def test_identical_descriptors_REAL(self):
        matcher = matchers.LoweRatioMatcher(ratio=.75)
        _template_test_identical_descriptors(matcher, IDENTICAL_REAL_DESCRIPTORS)

    def test_identical_descriptors_BINARY(self):
        matcher = matchers.LoweRatioMatcherBinary(ratio=.8)
        _template_test_identical_descriptors(
            matcher, IDENTICAL_BINARY_DESCRIPTORS_UINT8
        )
