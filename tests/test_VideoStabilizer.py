#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
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
WEBCAM_ID = 1  #  THIS COULD BE DIFFERENT ON EVERY SYSTEM, MAKE SURE THIS WEBCAM EXISTS
#NOTE: ALSO MAKE SURE that the webcam is seeing something useful, and not just a solid
# color or so. It will cut out PANEL from the full field of view.
# (which is purely my laziness, else I'd have to reparametrize the test again to make 
# the distinction between streaming from a cam and streaming from a video.)

OUTPUT_PATH = Template('figures/vidstab_test_stream_${stream}_pipeline_${pipeline}_layout_${layout}_${panel}')
START = 0; END = 1.0
vid_FPS = 5  #  synthetic vid is at 5FPS
cam_FPS = 30 #  setting cam FPS explicitly
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
@pytest.mark.parametrize("video_fps", [(VIDEO_PATH, vid_FPS, 'File'), (WEBCAM_ID, cam_FPS, 'Cam')], ids=['VideoFile', 'WebCam'])
@pytest.mark.parametrize("pipeline", [(None,'None'), (CUSTOM_PIPELINE,'custom')],ids=["default_pipeline","custom_pipeline"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("panel2", PANEL2 )
def test_video_stabilizer(
    video_fps, pipeline, layout, panel2
):
    outfile = os.path.join(
            TEST_FOLDER_ABSPATH, OUTPUT_PATH.substitute(
                pipeline=pipeline[1],
                layout=layout,
                panel=panel2,
                stream=video_fps[2])
            )
    vs = VideoStabilizer(
        video_fps[0], start=START, end=END, panel=PANEL, FPS=video_fps[1],
        panel2=panel2,
        pipeline=pipeline[0],
        output_layout=layout,
        video_outfile=outfile+'.mp4',
        numpy_outfile=outfile+'.npz'
    )
    vs.loop()
