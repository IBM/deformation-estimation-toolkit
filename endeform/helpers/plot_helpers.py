#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
import copy
import warnings
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


#%%
def mask_to_rgba(mask, alpha=0.3, mask_color="g"):
    # color may be anything that matplotlib understands as a color
    color = to_rgb(mask_color)
    gm_rgba = np.zeros(mask.shape + (4,))
    gm_rgba[mask, 3] = alpha
    gm_rgba[:, :, :3] = color
    return gm_rgba


#%%
def _fix_channel_order(img, channel_order="bgr"):
    """returns an indexer `idx` such that `img[:,:,idx]` is an RGB image, or,
    if img is only 2D, `img[:,:,idx]` still works and is identical to `img`"""
    if img.ndim == 2 or channel_order.lower() == "rgb":
        return Ellipsis
    elif img.ndim == 3 and channel_order.lower() == "bgr":
        return slice(None, None, -1)
    else:
        raise ValueError(
            f"You supplied either an unknown channel_order ({channel_order}) or an image with neither 2 nor 3 channels (img.shape={img.shape})"
        )


#%%
def overlay_mask_rgba(
    img,
    mask,
    alpha=0.3,
    mask_color="g",
    ax=None,
    mask_label="Mask",
    channel_order="bgr",
):
    # overlays the mask `mask` over `img` by converting `mask` to an rgba image
    if ax is None:
        _, ax = plt.subplots()
    imghandle = ax.imshow(img[:, :, _fix_channel_order(img, channel_order)])
    mask_rgba = mask_to_rgba(mask, alpha=alpha, mask_color=mask_color)
    maskhandle = ax.imshow(mask_rgba)
    if mask_label is not (None or ""):
        # create a fake line for the label
        # proxy = plt.Line2D([],[], color=to_rgb(mask_color)+(alpha,))
        proxy = Patch(color=to_rgb(mask_color) + (alpha,))
        ax.legend((proxy,), (mask_label,))
    else:
        proxy = None
    return ax, imghandle, maskhandle, proxy


def overlay_mask_cmap(
    img,
    mask,
    alpha=0.3,
    mask_color=None,
    ax=None,
    mask_label=None,
    cmap="Greens_r",
    boolean_mask=None,
    channel_order="bgr",
):
    """Overlay mask using colormap
    If `mask` contains float values, then `boolean_mask` has to be specified, too:
    `boolean_mask[y,x]==True` means the pixel at [y,x] is completely transparent,
    otherwise it's colored according to the value `mask[y,x]`.
    Specify either `mask_color` OR `cmap`. The former will override the latter.
    It is probably a good idea to use a reversed colormap, especially if the mask is Boolean, since if there is only a
    single value in the unmasked part, then only the lowest color of the colormap is used, and that is often white.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if mask_color is not None:
        #  Generate a colormap with a single color only.
        my_cmap = plt.cm.colors.ListedColormap([mask_color])
    elif isinstance(cmap, str):
        my_cmap = copy.copy(plt.cm.get_cmap(cmap))
    else:  #  Assume that cmap is a colormap already
        my_cmap = cmap
    my_cmap.set_bad(alpha=0.0)

    if mask_label is not None:
        print(
            "[Warning] Label and legend entry for the mask is not implemented in this function. \
                   Either use overlay_mask_rgba, or add the label manually."
        )

    if boolean_mask is None:
        boolean_mask = mask

    # convert the masked to a masked array.
    # plt will treat the elements of a masked array where the mask is true as "bad"
    masked_mask = np.ma.masked_array(data=mask, mask=~boolean_mask)
    imghandle = ax.imshow(img[:, :, _fix_channel_order(img, channel_order)])
    maskhandle = ax.imshow(masked_mask, cmap=my_cmap, alpha=alpha)
    return ax, imghandle, maskhandle


#%%
def overlay_keypoints(
    img: np.array, kpts, marker="2", color="g", ax=None, channel_order="bgr"
):
    """overlays keypoints `kpts` on `img`, plots using matplotlib. Use img=None if you want only keypoints
    TODO: add options to indicate response strength, size, angle of keypoints
    """
    if ax is None:
        _, ax = plt.subplots()
    if img is not None:
        ax.imshow(img[:, :, _fix_channel_order(img, channel_order=channel_order)])
    ax.plot(
        [k.pt[0] for k in kpts],
        [k.pt[1] for k in kpts],
        marker=marker,
        color=color,
        linestyle="none",
    )
    return ax


#%%
def draw_matches(
    img1: np.array,
    img2: np.array,
    kpts1,
    kpts2,
    matches,
    marker="2",
    marker_unmatched=None,
    color="g",
    ax=None,
    channel_order="bgr",
    linestyles=("-", "--", ":", "-."),
    tile_as="row",
):
    if tile_as in ["row", "h", 1]:
        w_flag, h_flag = (1, 0)
        glue_ax = 1
    elif tile_as in ["col", "column", "v", 0]:
        w_flag, h_flag = (0, 1)
        glue_ax = 0
    else:
        raise RuntimeError(f"{tile_as} is not a known option for `tile_as`")

    if ax is None:
        _, ax = plt.subplots()
    if img1 is not None:
        # glue and plot the images
        ax.imshow(
            np.concatenate(
                (
                    img1[:, :, _fix_channel_order(img1, channel_order=channel_order)],
                    img2[:, :, _fix_channel_order(img2, channel_order=channel_order)],
                ),
                axis=glue_ax,
            )
        )
    lstyles = cycle(linestyles)
    h, w, *_ = img1.shape
    for m in matches:
        x1, y1 = kpts1[m.queryIdx].pt
        x2, y2 = kpts2[m.trainIdx].pt
        ax.plot(
            (x1, x2 + w * w_flag),
            (y1, y2 + h * h_flag),
            marker=marker,
            color=color,
            linestyle=next(lstyles),
        )
    return ax


# %%
#%%
def rect_grid(X, Y, skip=(1, 1), ax=None, color="r", orientation="image", **kwargs):
    """Draws the rectangular grid specified by (X,Y) into ax (if given)
    `orientation` is either 'image', 'yx', or 'xy'.
     If it is 'xy', then the first array and first dimenstion is taken to correspond to the x-direction (like in a regular plot)
    otherwise it is taken to correspond to the y-direction (like in an image).
    Use `skip = (d1, d2)` to plot only every di-th line in the i direction.
    TODO: If `ax` is not given, plt.gca() is used if any figures exist. It would be better to check for an existing axes instead of figure.
          If no figures are open, a new one is created and if orientation is 'yx', the yaxis is then inverted.
    """
    if orientation.lower() not in ["yx", "image", "xy"]:
        warnings.warn(
            "Value {} for `orientation` argument unknown, assuming 'yx'".format(
                orientation
            ),
            RuntimeWarning,
        )
        orientation = "yx"
    if orientation.lower() in ["yx", "image"]:
        X, Y = Y, X
    elif orientation.lower() in ["xy"]:
        pass

    if ax is None:
        if plt.get_fignums():  #  if any figures exist
            ax = plt.gca()
        else:
            _, ax = plt.subplots()
            if orientation.lower() in ["yx", "image"]:
                #  also invert the y-axis direction in the new axis
                ax.invert_yaxis()

    # Note: The variable names below are confusing if the orientation is 'yx', but it's not worth refactoring.
    w, h = X.shape
    for horz in range(0, w, skip[0]):
        ax.plot(X[horz, :], Y[horz, :], color, **kwargs)
    if horz != w - 1:
        ax.plot(X[w - 1, :], Y[w - 1, :], color, **kwargs)
    for vert in range(0, h, skip[1]):
        ax.plot(X[:, vert], Y[:, vert], color, **kwargs)
    if vert != h - 1:
        ax.plot(X[:, h - 1], Y[:, h - 1], color, **kwargs)
    return ax
