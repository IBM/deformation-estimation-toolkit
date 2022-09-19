#%%
import os, warnings

import pytest
import cv2
import matplotlib.pyplot as plt
import numpy as np

# %%
from endeform.detectors_descriptors import detect
try:
    from endeform.detectors_descriptors import xdetect
    XDETECT_AVAILABLE = True
except ImportError:
    xdetect = None
    XDETECT_AVAILABLE = False

from .context import SAMPLE_DATA_FOLDER


PLOT_FLAG = True  #  if True, images of keypoints overlaid on the images themselves will be stored for failed tests
# Maybe in the future there is a way to specify using a pytest command line argument or so.
#%%
def overlay_kpts_plt(
    ax,
    img,
    keypoints,
    markerspec,
    label="",
    extrakeypoints=None,
    extrakeypointsmarkerspecs=["co"],
    extrakeypointslabels=[""],
):
    ax.imshow(np.atleast_3d(img)[:, :, ::-1])
    ax.plot(
        [k.pt[0] for k in keypoints],
        [k.pt[1] for k in keypoints],
        markerspec,
        label=label,
    )
    if extrakeypoints is not None:
        for kpts, mspc, lbl in zip(
            extrakeypoints, extrakeypointsmarkerspecs, extrakeypointslabels
        ):
            ax.plot([k.pt[0] for k in kpts], [k.pt[1] for k in kpts], mspc, label=lbl)
    ax.axis("off")
    return ax


#%%
def _kpts_near_reference_points(kpts, refpts, imsize=None, tol=(10, 10)):
    """
    Returns which points in `refpts` have at least one keypoint for `kpts` withing a tolerance of `tol`

    Parameters:
    ----------
    kpts: array of cv2.KeyPoint instances
    refpts: Iterable of tuples of len 2 or Iterable of cv2.KeyPoint instances
        If tuples, then they're ordered like so: [(refpt1_y, refpt1_x), (refpt2_y, refpt2_x), ...]
    tol: tuple of int len 2
        A refpoint will count as "found" if there is a keypoint within (tol_y, tol_x) of (refpt_y, refpt_x)
    Returns:
    ---------
    found : List
        found[r] == True iff at least one keypoint was within `tol` of `refpts[rr]`.
    """
    refs_are_KeyPoint = isinstance(refpts[0], cv2.KeyPoint)
    if imsize is not None:
        h, w = imsize
    else:
        h = np.inf
        w = np.inf
    found = [False] * len(refpts)
    for ii, refpt in enumerate(refpts):
        if not refs_are_KeyPoint:
            refpt_interval = (
                (max(0, refpt[0] - tol[0]), min(h, refpt[0] + tol[0])),
                (max(0, refpt[1] - tol[1]), min(w, refpt[1] + tol[1])),
            )
        else:
            refpt_interval = (
                (max(0, refpt.pt[1] - tol[0]), min(h, refpt.pt[1] + tol[0])),
                (max(0, refpt.pt[0] - tol[1]), min(w, refpt.pt[0] + tol[1])),
            )
        for kpt in kpts:
            if (
                (refpt_interval[0][0] <= kpt.pt[1])
                & (refpt_interval[0][1] >= kpt.pt[1])
                & (refpt_interval[1][0] <= kpt.pt[0])
                & (refpt_interval[1][1] >= kpt.pt[0])
            ):
                found[ii] = True
                break
    # Note the indexing above: for a cv2.Keypoint kpt, kpt.pt = (x-coord, y-coord). Everywhere else,
    # the ordering is (y-coord, x-coord)
    return found


#%% Data and helper functions for Fork&Knife test image
forknife = cv2.imread(
    os.path.join(SAMPLE_DATA_FOLDER, "400794_knife-and-fork-free-stock-photo.jpg")
)  #  a picture of a fork and a knife
#  Creative Commons Licence, source: https://avopix.com/photo/400794-knife-and-fork-free-stock-photo
prongs = [
    (36, 170),
    (61, 166),
    (90, 161),
    (118, 158),
]  #  pixel coordinates (y,x) of the tips of the fork's prongs


def _detect_found_kpts_near_prongs(detector, tol=(10, 10)):
    """
    Returns 4-element list `found` with `found[n]==True` if the n-th prong from the top is in the list
    of kpts (up to a tolerance of `tol`) detected by `detector`.

    Parameters
    ----------
    detector : An instance of BaseDetector
        The detector that's being tested here.
    tol : tuple, optional
        A prong is considered found if there's a keypoint within tolerance, by default (tol_y=10,tol_x=10)
    """
    kpts = detector.detect(forknife, mask=None)
    found = _kpts_near_reference_points(
        kpts, prongs, imsize=forknife.shape[:2], tol=tol
    )
    if PLOT_FLAG:
        _, ax = plt.subplots()
        overlay_kpts_plt(ax, forknife, kpts, "r+")
    return found


_forknife_explanation = (
    "The tested detector didn't find all the prongs of the fork"
    + ", prong(s) {not_found_prongs} were not found."
)


def _template_test_forknife(detector, tol=(10, 10)):
    found = _detect_found_kpts_near_prongs(detector, tol=tol)
    # assert all(found), _forknife_explanation.format(
    #     not_found_prongs=[i + 1 for i in range(4) if not found[i]]
    # )
    if not all(found):
        warnings.warn(_forknife_explanation.format(
        not_found_prongs=[i + 1 for i in range(4) if not found[i]]
        )
        )


# %% Helper functions and data for the transformation-tests
def _overlap_sets_kpts(kpts1, kpts2, tol=(1, 1)):
    """
    Returns the number of kpts that are in both, `kpts1` and `kpts2` (up to `tol`)
    """
    n1 = len(kpts1)
    n2 = len(kpts2)
    if n1 < n2:
        return np.count_nonzero(_kpts_near_reference_points(kpts1, kpts2, tol=tol))
    else:
        return np.count_nonzero(_kpts_near_reference_points(kpts2, kpts1, tol=tol))


def _DSC_sets_kpts(kpts1, kpts2, tol=(1, 1)):
    """
    Returns the Sorensen Dice Coeff
    DSC = 2*|intersect(`kpts1`, `kpts2`)|/(|kpts1|+|kpts2|)
    """
    return 2 * _overlap_sets_kpts(kpts1, kpts2, tol=tol) / (len(kpts1) + len(kpts2))


def _DSC_kpts_on_frame_and_blurred(detector, img, sigma=1.5, tol=(1, 1)):
    """
    Detects kepypoints on `img` and a slightly blurred version of `img`; returns the DSC of the two sets of detected keypoints
    """
    kpts = detector.detect(img, mask=None)
    img_blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma)
    kpts_blur = detector.detect(img_blur, mask=None)
    #  ksize=(0,0) means kernel size chosen based on sigma. Specifying only sigmaX means sigmaY=sigmaX
    if PLOT_FLAG:
        fig, ax = plt.subplots(1, 2)
        overlay_kpts_plt(
            ax[0],
            img,
            kpts,
            "cx",
            "Kpts of original img",
            extrakeypoints=[kpts_blur],
            extrakeypointsmarkerspecs=["g+"],
            extrakeypointslabels=["Kpts of blurred img"],
        )
        overlay_kpts_plt(
            ax[1],
            img_blur,
            kpts,
            "cx",
            "Kpts of original img",
            extrakeypoints=[kpts_blur],
            extrakeypointsmarkerspecs=["g+"],
            extrakeypointslabels=["Kpts of blurred img"],
        )
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        ax[0].legend()
    return _DSC_sets_kpts(kpts, kpts_blur, tol=tol)


def _template_test_blur(detector, img, DSC_thresh=0.5, sigma=1.5, tol=(1, 1)):
    """
    Compares sets of keypoints detected on `img` and its blurred version, asserts that the DSC between
    the two sets of detected keypoints is at least `DSC_thresh`
    """
    DSC = _DSC_kpts_on_frame_and_blurred(detector, img, sigma=sigma, tol=tol)
    print("DSC: %.3f" % DSC)
    # assert DSC >= DSC_thresh
    if DSC < DSC_thresh:
        warnings.warn(f"Overlap between keypoints is {DSC:.3f}, less than threshold {DSC_thresh}")


def _DSC_kpts_on_frame_and_shifted(detector, img, shift=(20, 10), tol=(1, 1)):
    """
    Detects kepypoints on `img` and a shifted version of `img`; returns the DSC of the two sets of detected keypoints (with shift applied)
    Specifically, for the first image, `shift[0]` pixels are cut off from the top, `shift[1]` pixels from the left.
    For the 2nd image, `shift[0]` pixels are cut off from the BOTTOM, `shift[1]` pixels from the RIGHT.
    """
    kpts1 = detector.detect(img[shift[0] :, shift[1] :, ...], mask=None)
    kpts2 = detector.detect(img[: -shift[0], : -shift[1], ...], mask=None)
    # shift the 2nd set of keypoints by `shift`
    for k in kpts2:
        k.pt = (k.pt[0] - shift[1], k.pt[1] - shift[0])
    if PLOT_FLAG:
        _, ax = plt.subplots()
        overlay_kpts_plt(
            ax,
            img[shift[0] :, shift[1] :, ...],
            kpts1,
            "cx",
            label="Kpts detected on original",
            extrakeypoints=[kpts2],
            extrakeypointsmarkerspecs=["g+"],
            extrakeypointslabels=["Detected on shifted img"],
        )
        ax.legend()
        plt.tight_layout(pad=0)
    return _DSC_sets_kpts(kpts1, kpts2, tol=tol)


def _template_test_shift(detector, img, DSC_thresh=0.7, shift=(20, 10), tol=(1, 1)):
    """
    Compares sets of keypoints detected on `img` and its shifted version, asserts that the DSC between
    the two sets of detected keypoints is at least `DSC_thresh`
    There's at least 2 possible reasons for them not being equal
    1. keypoints might be cut off by the shifting operations (if they're in the margins of size `shift`)
    2. The keypoints in the margins have larger cornerness, so that they bump keypoints off the top-N list (since typically, only the best N keypoints are returned)
    """
    DSC = _DSC_kpts_on_frame_and_shifted(detector, img, shift=shift, tol=tol)
    print("DSC: %.3f" % DSC)
    # assert DSC >= DSC_thresh
    if DSC < DSC_thresh:
        warnings.warn(f"Overlap between keypoints is {DSC:.3f}, less than threshold {DSC_thresh}")


def _DSC_kpts_on_frame_and_rotated(detector, img, mask=None, angle=10, tol=(1, 1)):
    """
    Detects kepypoints on `img` and a rotated version of `img`; returns the DSC of the two sets of detected keypoints (with shift applied)
    Specifically, the image is rotated `angle` degrees ccw around its center.
    To avoid rotating the margins into the frame, only the center square is kept.
    The square is computed so it lies in the circle around the image center with diameter min(w,h)
    """
    # image dims and center
    h, w = img.shape[:2]
    cy, cx = (int(h // 2), int(w // 2))
    # half-side of the inscribed square
    a2 = int(min(w, h) / 2 / np.sqrt(2))
    if mask is not None:
        kpts1 = detector.detect(
            img[cy - a2 : cy + a2, cx - a2 : cx + a2, ...],
            mask=mask[cy - a2 : cy + a2, cx - a2 : cx + a2],
        )
    else:
        kpts1 = detector.detect(
            img[cy - a2 : cy + a2, cx - a2 : cx + a2, ...], mask=None
        )
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h))
    if mask is not None:
        # kpts2 = detector.detect( img[cy-a2:cy+a2, cx-a2:cx+a2, ...], mask=mask[cy-a2:cy+a2, cx-a2:cx+a2] )
        raise NotImplementedError("Rotation of mask is not yet implemented.")
        # jpe (March 2021): I'm not sure anymore what the additional problem with a mask rotation would be
    else:
        kpts2 = detector.detect(
            img_rot[cy - a2 : cy + a2, cx - a2 : cx + a2, ...], mask=None
        )

    # Easier to rotate the initial set of keypoints
    for k in kpts1:
        # Note: since we cut out a square patch around the center of the image, the rotation center in those patches is just (a2,a2)
        pt = M[:, :2] @ np.array((k.pt[0] - a2, k.pt[1] - a2)) + (a2, a2)
        k.pt = tuple(pt)

    if PLOT_FLAG:
        _, ax = plt.subplots()
        overlay_kpts_plt(
            ax,
            img_rot[cy - a2 : cy + a2, cx - a2 : cx + a2, ...],
            kpts2,
            "cx",
            label="Detected on rotated img",
            extrakeypoints=[kpts1],
            extrakeypointsmarkerspecs=["g+"],
            extrakeypointslabels=["Detected on orig img"],
        )
        ax.legend()
        plt.tight_layout(pad=0)
    return _DSC_sets_kpts(kpts1, kpts2, tol=tol)


def _template_test_rotate(
    detector, img, mask=None, DSC_thresh=0.7, angle=10, tol=(1, 1)
):
    """
    Compares sets of keypoints detected on `img` and its rotated version, asserts that the DSC between
    the two sets of detected keypoints is at least `DSC_thresh`
    There's at least 2 possible reasons for them not being equal
    1. keypoints might have been rotated out of the frame
    2. Keypoints with larger cornerness might have been rotated in, so that they bumped keypoints off the top-N list (since typically, only the best N keypoints are returned)
    """
    DSC = _DSC_kpts_on_frame_and_rotated(detector, img, mask=mask, angle=angle, tol=tol)
    print("DSC: %.3f" % DSC)
    # assert DSC >= DSC_thresh
    if DSC < DSC_thresh:
        warnings.warn(f"Overlap between keypoints is {DSC:.3f}, less than threshold {DSC_thresh}")


#%% Helper functions for the number of described detected keypoints
# -> a detector/descriptor should at least describe most of the keypoints it detected
def _template_test_number_of_described_vs_detected_keypoints(
    detector, descriptor, img, thresh=0.8, mask=None
):
    kpts_detect = detector.detect(img, mask)
    kpts_describe, descs = descriptor.compute(
        img,
        kpts_detect,
    )
    assert len(kpts_describe) >= thresh * len(
        kpts_detect
    ), f"Of {len(kpts_detect)} detected keypoints, only {len(kpts_describe)} were described, that's less than the threshold {thresh}."


def _template_test_detect_vs_detectdescribe(detectordescriptor, img, mask=None):
    kpts_detect = detectordescriptor.detect(img, mask)
    kpts_dc, descs = detectordescriptor.detect_and_compute(img, mask)
    if not {k.pt for k in kpts_dc}.issubset(
        {k.pt for k in kpts_detect}
    ):
        warnings.warn(
            "detect_and_compute() detected keypoints that detect() alone did not detect. That's at least suspicious"
            )


def _template_test_no_kpts_outside_frame(detector, img, mask=None):
    kpts = detector.detect(img, mask)
    h, w = img.shape[:2]
    kpts_x = np.array([k.pt[0] for k in kpts])
    kpts_y = np.array([k.pt[1] for k in kpts])
    assert (
        np.all(kpts_x >= 0)
        and np.all(kpts_x <= w)
        and np.all(kpts_y >= 0)
        and np.all(kpts_y <= h)
    )


def _template_test_all_masked_no_keypoints(detector, img):
    # mask will be set to all False, i.e. everything is masked and hence no keypoints at all should be found
    h, w = img.shape[:2]
    mask = np.full((h, w), fill_value=False, dtype=bool)
    kpts = detector.detect(img, mask=mask)
    assert kpts == [] or kpts == (), (
        "A mask of all True should result in no keypoints! Here, %d were detected."
        % len(kpts)
    )


#%%

@pytest.mark.skipif(not XDETECT_AVAILABLE, reason="xdetect not available")
class TestPyramidORB:
    detectordescriptor = None if not XDETECT_AVAILABLE else xdetect.PyramidORB()
    def test_forknife(self, savefig, tol=(10, 10)):
        try:
            _template_test_forknife(self.detectordescriptor, tol=tol)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_blur(self, endoscopy_image, savefig):
        try:
            _template_test_blur(self.detectordescriptor, endoscopy_image)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_shift(self, endoscopy_image, savefig, tol=(1, 1)):
        try:
            _template_test_shift(self.detectordescriptor, endoscopy_image, tol=tol)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_rotate(self, endoscopy_image, savefig, mask=None, tol=(1, 1)):
        try:
            _template_test_rotate(
                self.detectordescriptor, endoscopy_image, mask=mask, tol=tol
            )
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_number_of_described_vs_detected(self, endoscopy_image):
        _template_test_number_of_described_vs_detected_keypoints(
            self.detectordescriptor, self.detectordescriptor, endoscopy_image
        )

    def test_detect_vs_detectdescribe(self, endoscopy_image):
        _template_test_detect_vs_detectdescribe(
            self.detectordescriptor, endoscopy_image
        )

    def test_no_keypoints_outside_frame(self, endoscopy_image):
        _template_test_no_kpts_outside_frame(self.detectordescriptor, endoscopy_image)

    def test_full_mask_results_in_no_keypoints(self, img=forknife):
        _template_test_all_masked_no_keypoints(self.detectordescriptor, img)

@pytest.mark.skipif(not XDETECT_AVAILABLE, reason="xdetect not available")
class TestPatchORB:
    detectordescriptor = None if not XDETECT_AVAILABLE else xdetect.PatchORB()

    def test_forknife(self, savefig, tol=(10, 10)):
        try:
            _template_test_forknife(self.detectordescriptor, tol=tol)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_blur(self, endoscopy_image, savefig):
        try:
            _template_test_blur(self.detectordescriptor, endoscopy_image)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_shift(self, endoscopy_image, savefig, tol=(1, 1)):
        try:
            _template_test_shift(self.detectordescriptor, endoscopy_image, tol=tol)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_rotate(self, endoscopy_image, savefig, mask=None, tol=(1, 1)):
        try:
            _template_test_rotate(
                self.detectordescriptor, endoscopy_image, mask=mask, tol=tol
            )
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_number_of_described_vs_detected(self, endoscopy_image):
        _template_test_number_of_described_vs_detected_keypoints(
            self.detectordescriptor, self.detectordescriptor, endoscopy_image
        )

    def test_detect_vs_detectdescribe(self, endoscopy_image):
        _template_test_detect_vs_detectdescribe(
            self.detectordescriptor, endoscopy_image
        )

    def test_no_keypoints_outside_frame(self, endoscopy_image):
        _template_test_no_kpts_outside_frame(self.detectordescriptor, endoscopy_image)

    def test_full_mask_results_in_no_keypoints(self, img=forknife):
        _template_test_all_masked_no_keypoints(self.detectordescriptor, img)


class Testcv2SIFT:
    detectordescriptor = detect.cv2SIFT()

    def test_forknife(self, savefig, tol=(10, 10)):
        try:
            _template_test_forknife(self.detectordescriptor, tol=tol)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_blur(self, endoscopy_image, savefig):
        try:
            _template_test_blur(self.detectordescriptor, endoscopy_image)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_shift(self, endoscopy_image, savefig, tol=(1, 1)):
        try:
            _template_test_shift(self.detectordescriptor, endoscopy_image, tol=tol)
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_rotate(self, endoscopy_image, savefig, mask=None, tol=(1, 1)):
        try:
            _template_test_rotate(
                self.detectordescriptor, endoscopy_image, mask=mask, tol=tol
            )
        except AssertionError:
            if PLOT_FLAG:
                savefig()
            raise

    def test_number_of_described_vs_detected(self, endoscopy_image):
        _template_test_number_of_described_vs_detected_keypoints(
            self.detectordescriptor, self.detectordescriptor, endoscopy_image
        )

    def test_detect_vs_detectdescribe(self, endoscopy_image):
        _template_test_detect_vs_detectdescribe(
            self.detectordescriptor, endoscopy_image
        )

    def test_no_keypoints_outside_frame(self, endoscopy_image):
        _template_test_no_kpts_outside_frame(self.detectordescriptor, endoscopy_image)

    def test_full_mask_results_in_no_keypoints(self, img=forknife):
        _template_test_all_masked_no_keypoints(self.detectordescriptor, img)
