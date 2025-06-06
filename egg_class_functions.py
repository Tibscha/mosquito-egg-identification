import numpy as np
import pandas as pd
from skimage import transform, io, img_as_ubyte, filters
from skimage.util import pad

def segmentation(image, model):
    """
    Perform segmentation on the given image using the 
    specified napari-convpaint model.

    Args:
        image (numpy.ndarray): The input image to be segmented.
        model (object): The segmentation model to use.

    Returns:
        numpy.ndarray: The segmented output.
    """

    if image.shape[2] == 3:
        image = np.moveaxis(image, -1, 0)

    segment = model.segment(image)

    return segment


def measure_sharpness(image_gray):
    laplacian = filters.laplace(image_gray)
    return laplacian.var()


def gradient_sharpness(image_gray):
    edge = filters.sobel(image_gray)
    return np.mean(edge)


def resize_with_padding(image, target_width = 3088, target_height = 2076):
    """
    Skaliert ein Bild auf die maximale Gr��e innerhalb von target_width x target_height
    unter Beibehaltung des Seitenverh�ltnisses, und f�gt schwarzen Rand (Padding) hinzu.
    """
    original_height, original_width = image.shape[:2]
    target_aspect = target_width / target_height
    original_aspect = original_width / original_height

    # Berechne neue Gr��e, die ins Zielformat passt (mit ratio)
    if original_aspect > target_aspect:
        # Bild ist breiter -> Breite = target_width
        new_width = target_width
        new_height = int(target_width / original_aspect)
    else:
        # Bild ist h�her -> H�he = target_height
        new_height = target_height
        new_width = int(target_height * original_aspect)

    # Resize Bild auf neue Gr��e
    image_resized = transform.resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True)
    image_resized = image_resized.astype(np.uint8)  # Optional: uint8 R�ckwandlung

    # Padding berechnen
    pad_height = target_height - new_height
    pad_width = target_width - new_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Padding hinzuf�gen (schwarz)
    if image_resized.ndim == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    image_padded = np.pad(image_resized, padding, mode='constant', constant_values=0)

    return image_padded
