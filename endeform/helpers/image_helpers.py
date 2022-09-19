#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
def tiler(img, patch_size, overlap=(0, 0), mask=None):
    """A generator returning patches from `img` and optionally `mask` of specified size and overlap

    Parameters
    ----------
    img : np.array
        The image. Must have 2 dimensions at least, 3 if multichannel image, but should work with more than 3 dimensions, too
    patch_size : tuple of 2 ints, or int
        Size (height, weight) of the patches. May be single number for square patches. The last patch of each row and column might end up being smaller
    overlap : tuple of 2 ints or int, optional
        Pixels by which current patch overlaps with next, as tuple (extension in y-direction, extension in x-direction) , by default (0,0)
    mask : np.array, optional
        2-D array of same height and width as `img`, which will be tiled, too. By default None, in which case it is ignored.

    Yields
    -------
    (iy, ix) : tuple
        index of current tile.
    tile : np.array
        the current tile
    mask_tile : np.array
        if `mask` was not None, then the current tile in mask. Else this is just None.

    Examples
    --------
    >>> img = np.arange(0,126,10)[:,np.newaxis] + np.arange(0,126,20)[np.newaxis,:]
    >>> img = np.dstack((img,img,img))
    >>> fig, ax = plt.subplots(7,7,squeeze=True,figsize=(15,10))
    >>> for ij,ptch,_ in tiler(img, patch_size=(5,3), overlap=(3,2)):
    >>>     ax[ij[1],ij[0]].imshow(ptch)
    >>>     ax[ij[1],ij[0]].set_title(ij)
    """
    try:
        p_y, p_x = patch_size
    except TypeError:
        p_y = p_x = patch_size
    try:
        o_y, o_x = overlap
    except TypeError:
        o_x = o_y = overlap
    h, w = img.shape[:2]

    bdry_y = bdry_x = iy = ix = 0
    while bdry_x < w:
        bdry_y = iy = 0
        while bdry_y < h:
            if mask is None:
                yield (iy, ix), img[
                    bdry_y : bdry_y + p_y, bdry_x : bdry_x + p_x , ...
                ], None
            else:
                yield (iy, ix), img[
                    bdry_y : bdry_y + p_y, bdry_x : bdry_x + p_x , ...
                ], mask[bdry_y : bdry_y + p_y, bdry_x : bdry_x + p_x]
            bdry_y += (p_y - o_y)
            iy += 1
        bdry_x += (p_x - o_x)
        ix += 1


def tiler_iterable(imgs, patch_size, overlap=(0, 0), height_width=None):
    """Same as `tiler`, but accepts an Iterable of images and no mask

    Parameters
    ----------
    imgs : Iterable of np.array
        The images to be tiled, and they should all have the same height
    patch_size : tuple of 2 ints, or int
        Size (height, weight) of the patches. May be single number for square patches. The last patch of each row and column might end up being smaller
    overlap : tuple of 2 ints or int, optional
        Pixels by which current patch overlaps with next, as tuple (extension in y-direction, extension in x-direction) , by default (0,0)
    height_width :  Tuple of int, optional
        The height and width of the supplied images, by default None, which means
        height and width of `imgs[0]` are used.

    Yields
    -------
    (iy, ix) : tuple
        index of current tile.
    tiles : generator
        yields the current tile from each image in `imgs`

    Examples:
    ---------
    imgs = (np.random.randint(255,size=(14,20,3),dtype=np.uint8),
                np.random.randint(255,size=(14,20),dtype=np.uint8))
    fig, ax = plt.subplots(2,4)
    for ij, ptch in tiler_iterable(imgs, patch_size=(8,10)):
        patch = tuple(ptch)  #  since `ptch` is a generator
        ax[ij[0],ij[1]].imshow(patch[0])
        ax[ij[0],ij[1]+2].imshow(patch[1])
        ax[ij[0],ij[1]].set_title(ij)
        ax[ij[0],ij[1]+2].set_title(ij)
    """
    try:
        p_y, p_x = patch_size
    except TypeError:
        p_y = p_x = patch_size
    try:
        o_y, o_x = overlap
    except TypeError:
        o_x = o_y = overlap

    if height_width is None:
        h, w = imgs[0].shape[:2]
    else:
        h, w = height_width  #  unpack

    bdry_y = bdry_x = iy = ix = 0
    while bdry_x < w:
        bdry_y = iy = 0
        while bdry_y < h:
            yield (iy, ix), (
                img[bdry_y : bdry_y + p_y, bdry_x : bdry_x + p_x, ...]
                for img in imgs
            )
            bdry_y += (p_y - o_y)
            iy += 1
        bdry_x += (p_x - o_x)
        ix += 1
