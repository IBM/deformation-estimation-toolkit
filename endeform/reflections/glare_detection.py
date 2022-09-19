import cv2
import numpy as np
from scipy.ndimage import convolve1d

from scipy.signal import find_peaks  


def mask_morphology(
    feature,
    threshold=None,
    smoothing_kernel=np.hanning(11) / 5,
    ASF_sizes=(5, 3, 3, 1),
    dilations=(5, 3),
    min_threshold=215,
    coeff_tophat=1.4,
    tophat_kernel=6,
):
    """
    Common part for the different glare detection functions,
    takes the feature image as input and computes the binary mask
    the other arguments are the one called in the different functions

    Notes
    -----
    Implementation based on the main principle of _[1]: it takes a feature image
    as input (computed in either `green_glare_mask` or `specularity_glare_mask`)
    and computes a binary mask of the big saturated and small bright regions
    that are supposed to correspond to glares.

    References
    ----------
    .. [1] Holger  Lange.   Automatic  glare  removal  in  reflectance  imagery
        of  the  uterine  cervix. Progress  in Biomedical Optics and Imaging
        - Proceedings of SPIE, 5747, 04 2005.

    """

    for size in ASF_sizes:
        if size % 2 == 0:
            raise ValueError(
                "ASF kernel sizes should be odd numbers (if not, a shifting of pixels"
                " is observed)."
            )

    # The shape of the structuring element used for the ASF filtering steps
    ASF_struct_elem_type = cv2.MORPH_ELLIPSE
    #  no pixel with lower feature value will be considered as a glare
    MIN_FEATURE = 125

    ## Finding big saturated regions
    # Histogram
    histogram = np.bincount(np.ravel(feature), minlength=256)
    # Smoothed histogram
    smoothed_histo = convolve1d(1.0 * histogram, smoothing_kernel, mode="reflect")
    # Using "reflect" avoids the excessive smoothing of peaks at the very ends of the
    #   histogram, here specifically near 0
    # the output is of the same type as the input, so with the 1.0 the smoothed histogram
    #   is quantized (rounded to int)

    # Adaptive threshold if no threshold given as input
    if threshold is None:
        # Minima
        # minima = peak_local_max(-smoothed_histo, indices=False, min_distance=10)
        a_min, _ = find_peaks(-smoothed_histo, distance=10, prominence=7)

        # If no peak could be found
        if a_min.size == 0:  # np.sum(minima) == 0:  # odd way of checking for empty
            threshold = min_threshold
        else:  # (general case)
            # Threshold based on the last minimum (most on the right)
            threshold = max(
                min_threshold, a_min.max()
            )  # Cannot be lower than 215 (arbitrary)

    # Saturated mask
    saturated_mask = (feature >= threshold).astype("uint8")

    ## Finding small bright regions
    # Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_kernel, tophat_kernel))

    # Top-hat transform and thresholding
    tophat = cv2.morphologyEx(feature, cv2.MORPH_TOPHAT, kernel)

    # Std-based thresholding
    _, small_bright_mask = cv2.threshold(
        tophat * (feature >= MIN_FEATURE),
        coeff_tophat * np.mean(tophat ** 2),
        255,
        cv2.THRESH_BINARY,
    )
    small_bright_mask = small_bright_mask.astype("uint8")

    ## ASF filtering on the glares
    # Kernels to use (bigger size -> bigger approximation)
    kernel1 = cv2.getStructuringElement(
        ASF_struct_elem_type, (ASF_sizes[0], ASF_sizes[0])
    )
    kernel2 = cv2.getStructuringElement(
        ASF_struct_elem_type, (ASF_sizes[1], ASF_sizes[1])
    )
    kernel3 = cv2.getStructuringElement(
        ASF_struct_elem_type, (ASF_sizes[2], ASF_sizes[2])
    )
    kernel4 = cv2.getStructuringElement(
        ASF_struct_elem_type, (ASF_sizes[3], ASF_sizes[3])
    )

    # Applying the ASF on the masks
    filtered_saturated = cv2.morphologyEx(
        cv2.morphologyEx(saturated_mask, cv2.MORPH_CLOSE, kernel1),
        cv2.MORPH_OPEN,
        kernel2,
    )
    filtered_small_bright = cv2.morphologyEx(
        cv2.morphologyEx(small_bright_mask, cv2.MORPH_CLOSE, kernel3),
        cv2.MORPH_OPEN,
        kernel4,
    )

    ## Dilation of the glares

    # Enlarging the big saturated regions (larger enlarging)

    enlarged_saturated_mask = cv2.dilate(
        filtered_saturated, np.ones((dilations[0], dilations[0]), np.uint8)
    )

    # Enlarging the small bright regions (smaller enlarging)
    enlarged_small_bright_mask = cv2.dilate(
        filtered_small_bright, np.ones((dilations[1], dilations[1]), np.uint8)
    )

    # Union of the two masks
    enlarged_mask = enlarged_saturated_mask | enlarged_small_bright_mask
    return enlarged_mask.astype("bool")


# Green channel-based function


def green_glare_mask(
    frame,
    threshold=None,
    smoothing_kernel=np.hanning(11) / 5,
    ASF_sizes=(5, 3, 3, 1),
    dilations=(5, 3),
    min_threshold=215,
    coeff_tophat=3.25,
    tophat_kernel=6,
    green_channel_index=1,
):
    """
    Parameters
    ----------
    frame : ndarray (h x w x 3)
        RGB image as an array
    threshold : int, optional
        threshold used on the glare feature image to consider pixels as part of
        a saturated region; adaptive if nothing is given
    smoothing_kernel : 1-D np.array
        histogram smoothing kernel; window size should be an odd number
        default: np.hanning(11)/5 (so it sums to 1)
    ASF_sizes : list of ints of len 4
        sizes of the kernels used in respectively:
        - closing on saturated regions
        - opening on closed saturated regions
        - closing on small bright regions
        - opening on closed small bright regions
        they should be odd numbers or shifting is observed
    dilations : list of ints of len 2
        sizes of the kernels used for the final dilations, respectively:
        - dilation of saturated regions
        - dilation of small bright regions
    min_threshold : int, optional
        lowest authorized threshold for the histogram-based adaptive threshold
        that is used to segment big saturated regions
    coeff_tophat : int or float, optional
        used for the thresholding of the top-hat feature map such that
        threshold = coeff_tophat*std(hist(tophat))
    tophat_kernel: int
        kernel size of the structuring element used in the tophat filtering step
    green_channel_index: {0,1,2}
        index of the green channel in `frame`. Typically (and default) 1

    Returns
    -------
    output : ndarray of bools (h x w x 1)
        binary mask of the detected glares. output[y,x]==True if pixel (x,y) is in a glare
    """

    ## Checks
    # if frame.ndim != 3:
    #     raise ValueError("There are not 3 dimensions, check if the input is RGB.")
    # if frame.shape[2] != 3:
    #     raise ValueError("There are not 3 channels, check if the input is RGB.")
    # Such checks are unpythonic I guess. We'll see an Exception below if anything was wrong.

    # Glare feature image as green channel
    feature = frame[:, :, green_channel_index]

    return mask_morphology(
        feature,
        threshold,
        smoothing_kernel,
        ASF_sizes,
        dilations,
        min_threshold,
        coeff_tophat,
        tophat_kernel,
    )


def specularity_glare_mask(
    frame,
    threshold=None,
    smoothing_kernel=np.hanning(11) / 5,
    ASF_sizes=(5, 3, 3, 1),
    dilations=(5, 3),
    min_threshold=215,
    coeff_tophat=1.4,
    tophat_kernel=6,
):
    r"""Computes a binary mask of glares in `frame`.

    The binary mask corresponds to the pixels that are considered as
    belonging to glares based on their spacularity, see _[2].
    The parametersthat are passed regulate the sensitivity of the detection.
    Small bright regions and saturated regions are detected independently
    (see function `glare_mask`).

    Parameters
    ----------
    frame : ndarray of shape (h,w,3)
        RGB image as an array
    threshold : int, optional
        threshold used on the glare feature image to consider pixels as part of
        a saturated region; adaptive if nothing is given
    smoothing_kernel : 1-D np.array
        histogram smoothing kernel; window size should be an odd number
        default: np.hanning(11)/5 (so it sums to 1)
    ASF_sizes : list of ints of len 4
        sizes of the kernels used in respectively:
        - closing on saturated regions
        - opening on closed saturated regions
        - closing on small bright regions
        - opening on closed small bright regions
        they should be odd numbers or shifting is observed
    dilations : list of ints of len 2
        sizes of the kernels used for the final dilations, respectively:
        - dilation of saturated regions
        - dilation of small bright regions
    min_threshold : int, optional
        lowest authorized threshold for the histogram-based adaptive threshold
        that is used to segment big saturated regions
    coeff_tophat : int or float, optional
        used for the thresholding of the top-hat feature map such that
        threshold = coeff_tophat*std(hist(tophat))
        coeff between 1 and 2 recommended

    Returns
    -------
    output : ndarray of bools (h x w x 1)
        binary mask of the detected glares

    Raises
    ------
    ValueError
        If the frame passed as an argument does not have the right dimensions

    Notes
    -----
        Computes a feature image based on the specularity component (see _[2]),
        then calls the function `glare_mask` with the other parameters to
        compute a binary mask of the glares from the feature image.

    References
    ----------
    .. [1] Holger  Lange.   Automatic  glare  removal  in  reflectance  imagery
        of  the  uterine  cervix. Progress  in Biomedical Optics and Imaging
        - Proceedings of SPIE, 5747, 04 2005.
    .. [2] Qingxiong Yang, Shengnan Wang, and Narendra Ahuja.  Real-time
        specular highlight removal usingbilateral filtering.  In Kostas Daniilidis,
        Petros Maragos, and Nikos Paragios, editors, Computer Vision– ECCV 2010,
        pages 87–100, Berlin, Heidelberg, 2010. Springer Berlin Heidelberg.
    """
    # Computing the chromaticities of the channels
    sum_channels = np.fmax(1, np.sum(frame, axis=2, keepdims=True))
    # ^ avoid 0 channel sums, which happens for all-black pixels
    sigma = frame / sum_channels

    sigma_min = np.nanmin(sigma, axis=2, keepdims=True)

    # Approximation of the diffuse chromaticity
    denominator = 1 - (3 * sigma_min)  # zero at pixels with r==b==g
    lambda_max = np.max(
        np.divide(
            sigma - sigma_min,
            denominator,
            out=np.ones_like(sigma),
            where=~np.isclose(denominator, 0),
        ),
        axis=2,
    )
    # Specular component
    feature = np.fmax(
        0,
        (np.max(frame, axis=2) - (lambda_max * sum_channels.squeeze()))
        / (1 - (3 * lambda_max)),
    ).astype(np.uint8)
    return mask_morphology(
        feature,
        threshold,
        smoothing_kernel,
        ASF_sizes,
        dilations,
        min_threshold,
        coeff_tophat,
        tophat_kernel,
    )
