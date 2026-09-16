import skimage.filters as skif


def _get_threshold(image, thresholding_algorithm):
    if thresholding_algorithm == "otsu":
        return skif.threshold_otsu(image)
    if thresholding_algorithm == "yen":
        return skif.threshold_yen(image)
    if thresholding_algorithm == "li":
        return skif.threshold_li(image)
    if thresholding_algorithm == "minimum":
        return skif.threshold_minimum(image)
    if thresholding_algorithm == "mean":
        return skif.threshold_mean(image)
    if thresholding_algorithm == "triangle":
        return skif.threshold_triangle(image)
    if thresholding_algorithm == "isodata":
        return skif.threshold_isodata(image)
    if thresholding_algorithm == "local":
        return skif.threshold_local(image)

    raise ValueError(
        f"Invalid thresholding algorithm: {thresholding_algorithm}"
    )


def threshold_image_multiregion(
    images,
    thresholding_algorithm,
    low_region_scale=0.8,
    high_region_scale=1.2,
):
    """
    Threshold the image stack into two named regions per image.

    Parameters
    ----------
    images : np.ndarray
        The image stack to threshold.
    thresholding_algorithm : str
        The thresholding algorithm to use.
    low_region_scale : float
        Multiplier applied to the base threshold for the first region.
    high_region_scale : float
        Multiplier applied to the base threshold for the second region.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two mask stacks corresponding to the mid-intensity and high-intensity
        regions.
    """
    base_threshold = _get_threshold(images, thresholding_algorithm)
    low_threshold = base_threshold * low_region_scale
    high_threshold = base_threshold * high_region_scale

    mid_masks = images > low_threshold
    high_masks = images > high_threshold

    return mid_masks, high_masks
