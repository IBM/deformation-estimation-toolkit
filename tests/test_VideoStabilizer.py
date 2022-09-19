import os
from string import Template

import pytest

from .context import SAMPLE_DATA_FOLDER, TEST_FOLDER_ABSPATH

from endeform.VideoStabilizer import VideoStabilizer
from endeform.pipeline import Pipeline
import endeform.detectors_descriptors.detect as ddd
from endeform.matchers.match import BruteForceMatcher
import endeform.interpolation.rigid as rigid

VIDEO_PATH = os.path.join(SAMPLE_DATA_FOLDER, 'synthetic_example.mp4')
#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
OUTPUT_PATH = Template('figures/vidstab_test_pipeline_${pipeline}_layout_${layout}_${panel}')
START = 0; END = 1.0
FPS = 5  #  synthetic vid is at 5FPS
PANEL = (0,0,389,256)
PANEL2 = [(0,256,389,256,[0,1,2]),(0,256,389,256,1), None, False]

LAYOUTS = ['1','2h','2v','4','6','9']

# custom pipeline
sift = ddd.cv2SIFT()
bfm = BruteForceMatcher()
aff = rigid.AffineInterpolator()

CUSTOM_PIPELINE = Pipeline(
    detector=sift,
    descriptor=sift,
    matcher=bfm,
    interpolator=aff
)

@pytest.mark.slow
@pytest.mark.parametrize("pipeline", [(None,'None'), (CUSTOM_PIPELINE,'custom')],ids=["default_pipeline","custom_pipeline"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("panel2", PANEL2 )
def test_video_stabilizer(
    pipeline, layout, panel2
):
    outfile = os.path.join(
            TEST_FOLDER_ABSPATH, OUTPUT_PATH.substitute(
                pipeline=pipeline[1],
                layout=layout,
                panel=panel2)
            )
    vs = VideoStabilizer(
        VIDEO_PATH, start=START, end=END, panel=PANEL, FPS=FPS,
        panel2=panel2,
        pipeline=pipeline[0],
        output_layout=layout,
        video_outfile=outfile+'.mp4',
        numpy_outfile=outfile+'.npz'
    )
    vs.loop()
