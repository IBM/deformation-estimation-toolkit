import os, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2

from .IO import ThreadedVideoStream as TVS
#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
from .pipeline import Pipeline
from .helpers import utils
from .detectors_descriptors import detect as ddd
from .matchers import match
from .filters import match_filters as mfilters
from .interpolation import TPS
# from .interpolation import rigid
from .reflections import glare_detection as glare


#%%
def overlay_query_points(img, match_list, keypoints):
    img0 = img.copy()
    for m in match_list:
        cv2.drawMarker(img0, tuple(map(int,keypoints[m.queryIdx].pt)), (0,255,0), cv2.MARKER_SQUARE, 1, cv2.LINE_4, 1)
    return img0

def overlay_train_points(img, match_list, keypoints):
    img0 = img.copy()
    for m in match_list:
        cv2.drawMarker(img0, tuple(map(int,keypoints[m.trainIdx].pt)), (0,255,0), cv2.MARKER_SQUARE, 1, cv2.LINE_4, 1)
    cv2.putText(img0, f'#matches={len(match_list)}', (300,350), cv2.FONT_HERSHEY_COMPLEX_SMALL, .8, (0,255,0))
    return img0

def resize_image_to_max(img, max_height_width):
    mh, mw = max_height_width
    h,w = img.shape[:2]
    fy = 1 if h<=mh else mh/h
    fx = 1 if w<=mw else mw/w
    f = min(fy,fx)
    return img if f>=1 else cv2.resize(img, None, None, f,f)


def assemble_output_frame6(vis1, grey1, vis1_warped, grey1_warped,
            vis0, initial_keypoints, last_matches, last_keypoints):
    """Note: all images are assumed to be 3-channel, so make sure they are before calling"""
    return    np.vstack((
        np.hstack( (vis1_warped, vis1,
                   overlay_query_points(vis0, last_matches, initial_keypoints)
                   ) ),
        np.hstack( (grey1_warped, grey1,
                   overlay_train_points(vis1, last_matches, last_keypoints)
                   ) )
        ))

def assemble_output_frame9(vis1, grey1, vis1_warped, grey1_warped,
            vis0, initial_keypoints, last_matches, last_keypoints, img_warped_grid):
    """Note: all images are assumed to be 3-channel, so make sure they are before calling"""
    h,w = vis1.shape[:2]
    return    np.vstack((
        assemble_output_frame6(vis1, grey1, vis1_warped, grey1_warped,
            vis0, initial_keypoints, last_matches, last_keypoints),
        np.hstack( (np.zeros((h, 2*w, 3), dtype=np.uint8),
                 img_warped_grid[:,:,::-1])
                    )
        ))

def assemble_output_frame2h(vis1, vis1_warped ):
    """Note: all images are assumed to be 3-channel, so make sure they are before calling"""
    return np.hstack( (vis1, vis1_warped))

def assemble_output_frame2v(vis1_warped, grey1_warped):
    """Note: all images are assumed to be 3-channel, so make sure they are before calling"""
    return np.vstack((vis1_warped, grey1_warped))

def assemble_output_frame4(vis1, grey1, vis1_warped, grey1_warped):
    """Note: all images are assumed to be 3-channel, so make sure they are before calling"""
    return np.vstack((
        np.hstack( (vis1_warped, vis1 ) ),
        np.hstack( (grey1_warped, grey1))
                    ))

# %% Video stabilizer

class VideoStabilizer:
    """A class wrapping the tedious setup and loop of frame acquisition, deformation
    estimation, deformation, and storage of video and potentially data.
    Example usage:

    >> vs = vidstab.VideoStabilizer('sample_data/synthetic_example.mp4',
        end=10.0,
        FPS=5,
        panel=(0,0,389,256), panel2=(0,256,389,256,1),
        output_layout='9',
        stats=True)
    >> vs.loop()

    which stabilize the first 10 seconds of the video. A video will be stored as
    'sample_data/synthetic_example.mp4' and the results will be available as numpy arrays
    in 'video_stabilized_data.npz'.

    See the Init docstring for more information.
    """
    def __init__(self,
        video, start:float=0.0, end:float=None, skip_frames:int=0,
        max_frames=None, FPS=None,
        min_matches=8, panel=None, panel2=None,
        store_numpy=True, numpy_outfile=None,
        video_outfile=None, output_layout='2h', out_grid=None,
        pipeline:Pipeline=None, interpolator=None, TPS_alpha:float=.1,
        lowe_ratio:float=.8,
        cam_stream_sleep=.5, resize_input_to=None
        ) -> None:
        """A summary

        Parameters
        ----------
        video : str or endeform.IO.ThreadedVideoStream
            The path to the video. May be relative and include '~'.
        start : float, optional
            The time, in seconds, at which to begin processing the video, by default 0.0
        end : float, optional
            The time, in seconds, at which to stop, by default None. If None, then the
            video will be processed until it ends. If both `max_frames` and `end` are given, `max_frames` takes precedence.
        skip_frames : int, optional
            Frames to skip when stabilizing, by default 0. This results in the FPS of the
            stabilized video being 1/(1+skip_frames) of the original video FPS, but will,
            of course, increase processing speed.
        max_frames : int, optional
            Maximum amout of frames to be processed, by default None. If both `max_frames` and `end` are given, `max_frames` takes precedence.
        FPS : float, optional
            It is difficult to obtain the FPS of a video from the video file directly, you
            can provide it here explicitly. It is used to convert `end` to a number of
            frames, and to set the FPS of the resulting video; by default None, in which 
            case the FPS are read from the cv2.VideoCapture object.
        min_matches : int, optional
            Processing will stop, if the current frame has less than `min_matches` with the
            initial frame, by default 15
        panel : tuple or list of (left,top,width,height), optional
            If specified, the video will be cropped to the corresponding rectangle before
            processing. That is helpful if the video is a composite view, or there are
            timestamps or similar metadata around the margins. If None, the whole video
            is processed, by default None
        panel2 : tuple or list of (left, top, width, height, channel(s) ), optional
            If speficied, this panel is stabilized using the deformation computed on the
            first panel. The data of this panel, stabilized, is stored as numpy files. If
            None, it's taken to be the same as the first panel, and if False, it is not
            used at all (no numpy data will be stored in this case), by default None.
        store_numpy : bool, optional
            Whether to store the resulting stabilized panel2 as a numpy file, by default True
        numpy_outfile : str, optional
            The filename for the numpy file. By default, it is be the name of the video
            + `_stabilized_data.npz`.
        video_outfile : str, optional
            The name of the produced output video, by default the name of the video +
            `_stabilized.mp4`.
        output_layout : str in {1,2h,2v,4,6,9}, optional
            The layout of the produced video, by default '2h'.
            Consider these "building blocks":
            F[t]  : the current frame (if `panel` is set, then only that section of it)
            cF[t] : the current frame, with applied estimated deformation
            F2[t], cF2[t]: same as above, but the 2nd panel (if `panel2` is set)
            kF0[t]: F[0] overlaid with the keypoints that were matched with keypoints in F[t]
            kF[t] : F[t] overlaid with the keypoints that were matched with keypoints in F[0]
            G[t]  : Regular grid, deformed by the estimated deformation between F[0] and F[t]
            Then the layouts are:
            '1'  :  [  cF[t] ]

            '2h' :  [ F[t]  cF[t]  ]

            '2v' :  [ cF[t]  ]
                    [ cF2[t] ]

            '4'  :  [ cF[t]   F[t]  ]
                    [ cF2[t]  F2[t] ]

            '6'  :  [ cF[t]   F[t]  kF0[t] ]
                    [ cF2[t]  F2[t]  kF[t] ]

            '9'  :  [ cF[t]   F[t]  kF0[t] ]
                    [ cF2[t]  F2[t]  kF[t] ]
                    [                 G[t] ]
            TODO: Add option for deformed grid spacing if part of layout
        out_grid : _type_, optional
            Not yet in use, by default None
        pipeline : Pipeline, optional
            The pipeline object used for the deformation estimation. By default, a pipeline
            of AKAZE detector, LATCH descriptor, Lowe matcher, and TPS is used.
        interpolator : , optional
            Not yet in use, by default None. Supply a Pipeline object if you want more
            freedom in the pipeline.
        TPS_alpha : float, optional
            If the default Pipeline is used, this is the alpha parameter for the TPS
            interpolator , by default .1
        lowe_ratio : float, optional
            The ratio in the default Lowe ration matcher, by default .8
        cam_stream_sleep : float
            If you're using a stream from a camera, there needs to be some time between 
            setting up and acquiring the first frame. If you experience problems with 
            completely black initial frames, try increasing this value. Default: .5 [seconds].
        resize_input_to : None or (w,h) tuple of int
            If set, the _input_ video is resized to (w,h) before any further processing.
            That means that `panel` and `panel2` are take from the _resized_ video. This
            can be helpful/necessary when dealing with cams of varying resolution.
        """
        if isinstance(video, str):
            self.video_path = os.path.abspath(os.path.expanduser(video))
            self.video_stream = self.cam_id = None
        elif isinstance(video, int) and video>=0:
            # camera id provided
            self.video_path = self.video_stream = None
            self.cam_id = video
        else:
            self.video_stream = video
            self.video_path = self.cam_id = None

        self.start = start
        self.end = end
        self.skip_frames = skip_frames
        self.max_frames = max_frames
        self.FPS = FPS
        self.panel = panel
        self.panel2 = panel2

        self.store_numpy = store_numpy
        self.numpy_outfile = numpy_outfile

        self.output_layout = output_layout
        self.video_outfile = video_outfile
        self.out_grid = out_grid # NOTE: Only None supported for now

        self.min_matches = min_matches
        self.pipeline = pipeline
        self.interpolator = interpolator
        self.TPS_alpha = TPS_alpha

        self.lowe_ratio = lowe_ratio

        self.extrapolation_color = [0,255,0]

        self._c2 = None # number of channels in "measurement" panel, panel2 (if it is specified)
        self._warped_grid = False
        
        self.__setup(cam_stream_sleep, resize_input_to)

    def __setup(self, cam_stream_sleep, resize_input_to):
        """ Default values where needed, and initialization steps."""

        # -- Video Stream
        # Figuring out what transform is needed on the read frames
        if self.panel is None:
            if self.panel2 is None or self.panel2 is False:
                frame_transform1 = lambda f: (f, None)
            else:
                raise ValueError("If `panel` is None, `panel2` can be only None or False.")
        else:
            x,y,w,h = self.panel
            if self.panel2 is None or self.panel2 is False:
               frame_transform1 = lambda f: (f[y:y+h,x:x+w,...], None)
            else:
                x2,y2,w2,h2,c2 = self.panel2
                if w2!=w or h2!=h:
                    raise ValueError("Panels have to have same size, but panel 1 "
                                     f"is {(w,h)} while panel2 is {(w2,h2)}")
                frame_transform1 = lambda f: (f[y:y+h,x:x+w,...], f[y2:y2+h,x2:x2+w,c2])

        if resize_input_to is None:
            frame_transform = frame_transform1
        else:
            frame_transform = lambda f: frame_transform1(cv2.resize(f, resize_input_to))

        if self.panel2 is None: self._c2 = 3
        elif self.panel2 is False: self._c2 = None
        else: self._c2 = getattr(self.panel2[4], '__len__', lambda: 1)()
        # this should return len() if it's available (i.e. a list or so was given) and
        # simply 1 if it's not (i.e. a scalar was supplied)
        
        # setup video stream
        CAM_STREAM = False
        if self.video_path is not None:
            self.video_stream = TVS.FileVideoStream(self.video_path, offset_ms=1000*self.start)
        elif self.video_stream is None:
            self.video_stream = TVS.CamVideoStream(self.cam_id, FPS=self.FPS)
            CAM_STREAM = True  #  so we can wait a beat after starting the stream
        # else we have a stream already

        if self.FPS is None:
            self.FPS = self.video_stream.get(cv2.CAP_PROP_FPS)
        if self.FPS is None:
            ## STILL TO DO. WITH CAMERA IT IS UNCLEAR WHAT THE REPORTED FPS MEAN IF WE
            ## DIDN'T SET THEM MANUALLY
            warnings.warn("Video Stream FPS not available, likely because you're using "
                "a camera and the backend doesn't provide the FPS. Setting FPS=30 to " 
                "avoid exceptions in further processing, but bear in mind that time is "
                "now unreliable.")
            self.FPS = 30

        if self.max_frames is None:
            if self.end is None:
                max_frames = None
            else:
                max_frames = int( (self.end - self.start)*self.FPS/(1+self.skip_frames) )
        else:
            max_frames = self.max_frames
        self.max_frames = max_frames

        self._cap = TVS.VideoStream(stream=self.video_stream,
                            transform=frame_transform,
                            skip_frames=self.skip_frames,
                            max_frames=max_frames,
                            default_if_empty=(None, None))
        if CAM_STREAM:
            time.sleep(cam_stream_sleep)                    
        self._cap.start()

        # acquire first frame
        vis00, grey00 = self._cap.read()
        if vis00 is None:
            raise RuntimeError("Acquisition of first frame failed. Did you provide "
                                f'a valid video? I have {self.video_path}')
        self.initial_frame = vis00
        self.initial_frame2 = grey00 #  could be None, and is likely not needed
        h,w = vis00.shape[:-1]
        self.w, self.h = (w,h)

        # -- set up the output video
        if self.video_outfile is None:
            self.video_outfile = os.path.splitext(self.video_path)[0] + '_stabilized.mp4'

        if self.out_grid is None:
            yy, xx = np.meshgrid(np.arange(self.h), np.arange(self.w))
            XY = np.column_stack( (xx.flat, yy.flat ) )
            self.out_grid = XY

        # Check if layout is compatible with output layout
        if self.panel2 is False and self.output_layout not in ('1','2h'):
            warnings.warn(f'Layout {self.output_layout} is incompatible with '
            '`panel2==False`. Changing to "2h"')
            self.output_layout='2h'

        if self.output_layout=='2h':
            self.__assemble_output = self.__assemble_output_2h
            self._output_frame_size = (2*w,h)
            self._warped_grid = False
        elif self.output_layout=='2v':
            self.__assemble_output = self.__assemble_output_2v
            self._output_frame_size = (w, 2*h)
            self._warped_grid = False
        elif self.output_layout=='4':
            self.__assemble_output = self.__assemble_output_4
            self._output_frame_size = (2*w, 2*h)
            self._warped_grid = False
        elif self.output_layout=='6':
            self.__assemble_output = self.__assemble_output_6
            self._output_frame_size = (3*w, 2*h)
            self._warped_grid = False
        elif self.output_layout=='9':
            self.__assemble_output = self.__assemble_output_9
            self._output_frame_size = (3*w, 3*h)
            self._warped_grid = True
        elif self.output_layout=='1':
            self.__assemble_output = self.__assemble_output_1
            self._output_frame_size = (w,h)
            self._warped_grid = False
        else:
            raise ValueError(f"Unknown output layout {self.output_layout}")

        self._writer = cv2.VideoWriter( self.video_outfile,
                cv2.VideoWriter_fourcc(*'avc1'),
                self.FPS//(self.skip_frames+1), self._output_frame_size, True)

        # -- set up pipeline
        if self.pipeline is None:
            akaze = ddd.cv2AKAZE(threshold=.0001) # detector descriptor
            latch = ddd.cv2LATCH() # descriptor only
            lowe = match.LoweRatioMatcherBinary(ratio=self.lowe_ratio)
            avg_trans = mfilters.average_translation_filter(factor=1.1)
            if self.interpolator is None or self.interpolator.upper()=='TPS':
                interpol = TPS.TPS(self.TPS_alpha)
            else:
                raise NotImplementedError(
                    "Only implemented TPS so far. Manually specify and supply Pipeline for more freedom")
            self.pipeline = Pipeline(
                    detector=akaze,
                    descriptor=latch,
                    keypoint_mask=glare.green_glare_mask,
                    matcher=lowe,
                    match_filter=avg_trans,
                    interpolator=interpol,
                    )

        # initialize pipeline
        self.pipeline.init(self.initial_frame)

        # -- storing numpy?
        if self.max_frames is None and self.store_numpy:
            warnings.warn('Storing results on disk is only possible if the number of '
            'frames is known. Either specify `max_frames` or `end`.')
            self.store_numpy = False
        if self.panel2 is False and self.store_numpy:
            warnings.warn('Storing results on disk is only possible if panel2 is '
            'specified')
            self.store_numpy = False
        if self.store_numpy:
            if self._c2 == 1:
                self.G = np.empty((h,w,self.max_frames), dtype=np.uint8)
                self.G[:,:,0] = grey00
            else:
                self.G = np.empty((h,w,self.max_frames, self._c2), dtype=np.uint8)
                self.G[:,:,0,...] = vis00 if self.panel2 is None else grey00
            self.G_valid = np.full((h,w,self.max_frames), True, dtype=bool)
            if self.numpy_outfile is None:
                self.numpy_outfile = os.path.splitext(self.video_path)[0] + '_stabilized_data.npz'

    def loop(self, return_arrays=False, stats=False, stats_every: int=60,
                live_view=False, live_view_window_name="Live View", live_update_every: int=1,
                live_view_max_height=9999, live_view_max_width=9999):
        """Starts the stabilization loop. Optionally displays some stats, and returns the
        numpy arrays corresponding to the stabilized values in panel2

        Parameters
        ----------
        return_arrays : bool, optional
            If True, returns `G`, `G_valid`, see below, else no value is returned, by default False.
            NOTE: Currently only available if `store_numpy` is True, because we pre-allocate
            the arrays, and without specifying (at least an upper bound on) the maximum
            number of frames, that isn't possible.
        stats : bool, optional
            If True, prints an update with FPS every `stats_every` frames, by default False
        stats_every : int, optional
            By default 60
        live_view : bool, optional
            If True, shows the produced video frame (according to `output_layout`) in an
            OpenCV GUI window and updates it every `live_view_every` frames.
            NOTE: The window should be resizeable, but often isn't. 
        live_view_every : int, optional
            By default 1
        live_view_window_name : str
            By default 'Live View'.
        live_view_max_height, live_view_max_width : int, optional.
            By default 9999. If given, resizes the image before updating the live view, 
            so that the height/width doesn't exceed the given value.
        Returns
        -------
        G : (h,w,N,c) or (h,w,N) np.array of uint8
            The values extracted from the stabilized `panel2`. N is the number of frames
            processed, and c is the number of channels. If c==1, the singleton dimension
            is removed.
        G_valid : (h,w,N) np.array of bool
            G_valid[y,x,f] == False, if pixel (x,y) in frame f corresponded to a location
            outside the frame (after deformation).
        """
        frame_ctr = 0; t00 = time.perf_counter(); t0 = t00
        num_matches = [0] if self.max_frames is None else np.zeros(self.max_frames, dtype=int)
        # matched_indices = [[]] if self.max_frames is None else \
        #                         [ [] for i in range(self.max_frames) ]

        if return_arrays and not self.store_numpy:
            warnings.warn('You did not set `store_numpy` to True, so we are not keeping'
            ' track of the arrays and will not be able to return them.')

        if live_view:
            cv2.namedWindow(live_view_window_name, flags=cv2.WINDOW_NORMAL)

        if stats: print('[INFO] Starting loop')
        while self._cap.isNotDone():
            # acquire next frame
            vis, grey = self._cap.read()

            if self.panel2 is None: grey = vis
            # estimate next deformation
            self.pipeline.step(vis)
            if len(self.pipeline._last_matches)<self.min_matches:
                premature = True
                warnings.warn(f'Stopping loop prematurely because frame {frame_ctr+1} '
                    f'has only {len(self.pipeline._last_matches)} matches, less than '
                    f' `min_matches` ({self.min_matches}).')
                break
            frame_ctr+=1

            # record some stats
            if self.max_frames is None:
                num_matches+=[len(self.pipeline._last_matches)]
                # matched_indices+=[[m.queryIdx for m in self.pipeline._last_matches]]
            else:
                num_matches[frame_ctr] = len(self.pipeline._last_matches)
                # matched_indices[frame_ctr] = [m.queryIdx for m in self.pipeline._last_matches]

            vis_warped, grey_warped, warped_grid = self._warp_imgs(vis,
                (vis if self.panel2 is None else
                 (None if self.panel2 is False else grey)
                ),
                self._warped_grid)

            output_frame = self.__assemble_output(
                vis, grey, vis_warped, grey_warped, warped_grid
                ).astype(np.uint8)

            self._writer.write(output_frame)

            if self.store_numpy:
                if self._c2==1:
                    self.G[~np.isnan(grey_warped),frame_ctr] = \
                        grey_warped[~np.isnan(grey_warped)].astype(np.uint8)
                    self.G_valid[np.isnan(grey_warped),frame_ctr] = False
                else:
                    self.G[
                        ~np.any(np.isnan(grey_warped), axis=2), frame_ctr,:
                        ] = grey_warped[
                            ~np.any(np.isnan(grey_warped), axis=2), ...
                        ].astype(np.uint8)
                    self.G_valid[
                        ~np.any(np.isnan(grey_warped), axis=2), frame_ctr
                        ] = False


            if stats and not (frame_ctr % stats_every):
                print(f'[INFO] Processed {frame_ctr} frames of '
                    f'{ "inf" if self.max_frames is None else self.max_frames }. '
                    f'FPS: {stats_every/(time.perf_counter()-t0)}.')
                t0 = time.perf_counter()
            
            if live_view and not (frame_ctr % live_update_every):
                cv2.imshow(live_view_window_name, resize_image_to_max(output_frame,
                                     (live_view_max_height, live_view_max_width)))
                keypress = cv2.waitKey(5) % 2**16  # need only least two bytes
                if keypress == ord('q') or keypress == ord('Q') or keypress == 27: # q or ESC to quit
                    print("[INFO] Pressed {} - processing stopped ".format(keypress))
                    break
                elif keypress == 65535:  # -1 % 2**16, and -1 is the default returned if no key is pressed
                    pass
                else: 
                    print(f'[INFO] Pressed {repr(chr(keypress))}, which means nothing to me. Press q or ESC to quit.')


        # Truncate to the number of actually processed frames
        if self.store_numpy:
            self.G = self.G[:,:,:frame_ctr,...]
            self.G_valid = self.G_valid[:,:,:frame_ctr]
        num_matches = num_matches[:frame_ctr]
        # save the numpy files
        if self.store_numpy:
            np.savez_compressed(self.numpy_outfile, M=self.G.squeeze(), M_valid=self.G_valid,
                initial_frame=self.initial_frame,
                num_matches=num_matches, offset=self.start,
                skip_frames=self.skip_frames, FPS=self.FPS)

        if stats: print(f'[INFO] Done. FPS: {frame_ctr/(time.perf_counter()-t00)}.')
        self._writer.release()
        self._cap.stop()
        if return_arrays:
            if self.panel2 is False or not self.store_numpy: return None, None
            else:
                return self.G.squeeze(), self.G_valid

    def _warp_imgs(self, img1, img2=None, warped_grid=False):
        """Warp all the required images"""
        Z = self.pipeline.eval(self.out_grid).reshape((self.w, self.h, 2))
        img_warped_stacked = utils.bilin_interp(
                Z[:,:,1].T, Z[:,:,0].T, np.arange(self.h), np.arange(self.w),
                np.dstack( (img1,) + (() if img2 is None else (img2,)) ),
                extrapolation_value=self.extrapolation_color + ([] if img2 is None else [np.nan]*self._c2)
                )

        if warped_grid:
            grid_fig, grid_ax = plt.subplots()
            self.pipeline._interpolator.draw_warped_grid(wh=(self.w, self.h), ax=grid_ax)
            plt.axis('equal'); plt.axis('off'); plt.tight_layout()
            canvas = grid_fig.canvas
            canvas.draw()
            img_warped_grid = cv2.resize( np.array(canvas.buffer_rgba())[:,:,:3],
                                                (self.w,self.h), interpolation=cv2.INTER_AREA)
            plt.close(fig=grid_fig)
        else:
            img_warped_grid = None
        return (img_warped_stacked[:,:,:3],
                    None if img2 is None else img_warped_stacked[:,:,3:].squeeze(),
                    img_warped_grid
            )

    def __assemble_output_1(self, vis1, grey1, vis1_warped, grey1_warped, warped_grid):
        return vis1_warped

    def __assemble_output_2h(self, vis1, grey1, vis1_warped, grey1_warped, warped_grid):
        return assemble_output_frame2h(vis1, vis1_warped)

    def __assemble_output_2v(self, vis1, grey1, vis1_warped, grey1_warped, warped_grid):
        return assemble_output_frame2v(vis1_warped,
            (grey1_warped if self._c2==3 else
            np.broadcast_to(grey1_warped.squeeze()[...,np.newaxis], (self.h, self.w, 3)) )
        )

    def __assemble_output_4(self,  vis1, grey1, vis1_warped, grey1_warped, warped_grid):
        return assemble_output_frame4(vis1,
            (grey1 if self._c2==3 else
            np.broadcast_to(grey1.squeeze()[...,np.newaxis], (self.h, self.w, 3))),
            vis1_warped,
            self.__replace_NaN_with_color(
                (grey1_warped if self._c2==3 else
        np.broadcast_to(grey1_warped.squeeze()[...,np.newaxis], (self.h, self.w, 3))
                ),
                color=self.extrapolation_color
                                        )
        )

    def __assemble_output_6(self,  vis1, grey1, vis1_warped, grey1_warped, warped_grid):
        return assemble_output_frame6(  vis1,
            (grey1 if self._c2==3 else
            np.broadcast_to(grey1.squeeze()[...,np.newaxis], (self.h, self.w, 3))),
            vis1_warped,
            (grey1_warped if self._c2==3 else
            np.broadcast_to(grey1_warped.squeeze()[...,np.newaxis], (self.h, self.w, 3))),
            self.initial_frame, self.pipeline.initial_keypoints,
            self.pipeline._last_matches, self.pipeline._last_keypoints)

    def __assemble_output_9(self,  vis1, grey1, vis1_warped, grey1_warped, warped_grid):
        return assemble_output_frame9( vis1,
            (grey1 if self._c2==3 else
            np.broadcast_to(grey1.squeeze()[...,np.newaxis], (self.h, self.w, 3))),
            vis1_warped,
            (grey1_warped if self._c2==3 else
            np.broadcast_to(grey1_warped.squeeze()[...,np.newaxis], (self.h, self.w, 3))),
            self.initial_frame, self.pipeline.initial_keypoints,
            self.pipeline._last_matches, self.pipeline._last_keypoints, warped_grid)

    def __replace_NaN_with_color(self, img, color):
        """replaces every pixel in the (h,w,3) array `img` for which at least one channel
        equals NaN by `color`"""
        img2 = np.zeros_like(img)
        img2[:,:,:] = np.array(self.extrapolation_color).reshape((1,1,3))
        img2[ np.all(~np.isnan(img), axis=2), ... ] = img[ np.all(~np.isnan(img), axis=2), ... ]
        return img2




# %%
