import numpy as np

#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
from endeform.reflections import glare_detection as gd


#%%
class Test_green_glare:
    def test_mask_is_bool(self, endoscopy_image):
        # This test simply asserts that the glare detection function returns an array of type bool
        mask = gd.green_glare_mask(endoscopy_image)
        assert np.issubdtype(mask.dtype, bool)

    def test_mask_is_same_shape(self, endoscopy_image):
        # This test asserts that the glare detection function returns an array of the same shape as the input image
        mask = gd.green_glare_mask(endoscopy_image)
        h, w = endoscopy_image.shape[:2]
        assert mask.shape == (h, w)
class Test_specularity_glare:
    def test_mask_is_bool(self, endoscopy_image):
        # This test simply asserts that the glare detection function returns an array of type bool
        mask = gd.specularity_glare_mask(endoscopy_image)
        assert np.issubdtype(mask.dtype, bool)

    def test_mask_is_same_shape(self, endoscopy_image):
        # This test asserts that the glare detection function returns an array of the same shape as the input image
        mask = gd.specularity_glare_mask(endoscopy_image)
        h, w = endoscopy_image.shape[:2]
        assert mask.shape == (h, w)
