import os
import skimage
import egg_class_functions as ecf
import numpy as np
import pandas as pd
from skimage import transform, io, img_as_ubyte, filters
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, erosion, dilation, footprint_rectangle, remove_small_holes
from skimage.util import img_as_ubyte


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
        new_width = int(target_width)
        new_height = int(target_width / original_aspect)
    else:
        # Bild ist h�her -> H�he = target_height
        new_height = int(target_height)
        new_width = int(target_height * original_aspect)

    # Resize Bild auf neue Gr��e
    image_resized = transform.resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True)
    image_resized = image_resized.astype(np.uint8)  # Optional: uint8 R�ckwandlung

    # Padding berechnen
    pad_height = int(target_height - new_height)
    pad_width = int(target_width - new_width)

    pad_top = int(pad_height // 2)
    pad_bottom = int(pad_height - pad_top)
    pad_left = int(pad_width // 2)
    pad_right = int(pad_width - pad_left)


    # Padding hinzuf�gen (schwarz)
    if image_resized.ndim == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    image_padded = np.pad(image_resized, padding, mode='constant', constant_values=0)

    mask = np.zeros((int(target_height), int(target_width)), dtype=bool)
    mask[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = True

    return image_padded, mask


def egg_image_data_import(image_paths):
    data = []
    images = []
    for path in image_paths:
        image = skimage.io.imread(path)
        
        path = path
        species = os.path.basename(os.path.dirname(path))
        category = os.path.basename(os.path.dirname(os.path.dirname(path)))
        data.append({
            'path': path,
            'category': category,
            'species': species
        })
        images.append(image)

    df = pd.DataFrame(data)
    return images, df


def egg_image_import(image_paths):
    images = []
    for path in image_paths:
        image = skimage.io.imread(path)
        images.append(image)
    return images

def region_sepperation(segment_mask):
    mask_cleaned = remove_small_holes(segment_mask == 2, area_threshold=5000)
    labeled_overlay = label(mask_cleaned)
    labeled_overlay = erosion(labeled_overlay, footprint=footprint_rectangle((25, 25)))
    labeled_overlay = remove_small_objects(labeled_overlay, min_size=20000)
    labeled_overlay = dilation(labeled_overlay, footprint_rectangle((25, 25)), mode='ignore')
    regions = regionprops(labeled_overlay)

    return regions, labeled_overlay


def region_processing(image, labeled_overlay, region):
    
    # cutting out the specified region out of an image
    minr, minc, maxr, maxc = region.bbox
    cropped_image = image[minr:maxr, minc:maxc]
    mask = labeled_overlay[minr:maxr, minc:maxc] == region.label
    masked_image = np.zeros_like(cropped_image)
    for c in range(cropped_image.shape[2]):
        masked_image[..., c] = cropped_image[..., c] * mask
    image_gray = skimage.color.rgb2gray(masked_image)

    # get measure parameters for each segment and saving them to a specified list
    angle = region.orientation
    area = region.area
    perimeter = region.perimeter
    roundness = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
    length = region.axis_major_length
    width = region.axis_minor_length
    ratio = length / width
    laplacian = ecf.measure_sharpness(image_gray)
    edge = ecf.gradient_sharpness(image_gray)
    
    data = {
        'angle': angle,
        'area': area,
        'perimeter': perimeter,
        'roundness': roundness,
        'length': length,
        'width': width,
        'len_wid_ratio': ratio,
        'laplacian' : laplacian,
        'edge': edge
    }

    return masked_image, mask, data


def save_image(save_path, image, name):
    skimage.io.imsave(f"{save_path}/{name}.png", img_as_ubyte(image))


def make_key(row):
    return f"image_{row['image_index']}_segment_{row['region_index']}.png"


def egg_segmentation(image_paths, seg_model, height, width, save_path=None):
    """Segmenting pictures of eggs for multiple pictures in given path.
    Usable for setting up a workflow and training models as well as in prduction.
    Returning a dataframe of segment data and the corresponding images, segments and masks.
    
    Args:
        image_paths (path): The path to the images.
        seq_model (object): The loaded model to segment the images.
        height (int): Specifying a certain image height.
        width (int): Specifying a certain image width.
        save_path (path): The path were segmented images are saved to. If not given images are not saved.

    Returns:
        df (pandas.dataframe): dataframe containg segment parameters important for classification.
        images (list): A list of the original images.
        segment_mask (list): A list of segmentation masks for original images.
        segments (list): A list of segmented images
        masks (list): A list of the segmented masks
    """
    images = []
    segment_masks = []
    segments = []
    masks = []
    data = []
    
    for i, path in enumerate(image_paths):
        image = skimage.io.imread(path)
        
        path = path
        species = os.path.basename(os.path.dirname(path))
        category = os.path.basename(os.path.dirname(os.path.dirname(path)))

        # adjust the image to specified size. ratio is kept missing pixels are padded
        if image.shape[:2] != (height, width):
            image, content_mask = ecf.resize_with_padding(image, width, height)
        else:
            content_mask = np.ones((height, width), dtype=bool)

        # calling segmentation function to get a segmentation mask
        segment_mask = ecf.segmentation(image, seg_model)
        if image.shape[:2] != (height, width):
            segment_mask[~content_mask] = 1

        # improving segmentation mask with various skimage methods
        mask_cleaned = remove_small_holes(segment_mask == 2, area_threshold=5000)
        labeled_overlay = label(mask_cleaned)
        labeled_overlay = erosion(labeled_overlay, footprint=footprint_rectangle((25, 25)))
        labeled_overlay = remove_small_objects(labeled_overlay, min_size=20000)
        labeled_overlay = dilation(labeled_overlay, footprint_rectangle((25, 25)), mode='ignore')
        regions = regionprops(labeled_overlay)
        
        images.append(image)
        segment_masks.append(labeled_overlay)

        # divide picture into segmented areas
        for j, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            cropped_image = image[minr:maxr, minc:maxc]
            mask = labeled_overlay[minr:maxr, minc:maxc] == region.label
            masked_image = np.zeros_like(cropped_image)
            for c in range(cropped_image.shape[2]):
                masked_image[..., c] = cropped_image[..., c] * mask
            image_gray = skimage.color.rgb2gray(masked_image)
            
            # saving image and corresponding mask for each segment
            segments.append(masked_image)
            masks.append(mask)

            if save_path != None:
                skimage.io.imsave(f"{save_path}/segment_images/{category[0]}_image_{i}_segment_{j}.png", img_as_ubyte(masked_image))
                skimage.io.imsave(f"{save_path}/segment_masks/{category[0]}_mask_{i}_segment_{j}.png", img_as_ubyte(mask))

            # get measure parameters for each segment and saving them to a specified list
            angle = region.orientation
            area = region.area
            perimeter = region.perimeter
            roundness = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
            length = region.axis_major_length
            width = region.axis_minor_length
            ratio = length / width
            laplacian = ecf.measure_sharpness(image_gray)
            edge = ecf.gradient_sharpness(image_gray)
            
            if save_path != None:
                data.append({
                    'path': path,
                    'device': category,
                    'species': species,
                    'image': i,
                    'segment': j,
                    'angle': angle,
                    'area': area,
                    'perimeter': perimeter,
                    'roundness': roundness,
                    'length': length,
                    'width': width,
                    'len_wid_ratio': ratio,
                    'laplacian' : laplacian,
                    'edge': edge
                })
            else:
                data.append({
                    'path': path,
                    'image': i,
                    'segment': j,
                    'angle': angle,
                    'area': area,
                    'perimeter': perimeter,
                    'roundness': roundness,
                    'length': length,
                    'width': width,
                    'len_wid_ratio': ratio,
                    'laplacian' : laplacian,
                    'edge': edge
                })

    df = pd.DataFrame(data)
    if save_path != None:
        df.to_csv(f"{save_path}/segmentation_data.csv")

    return df, images, segment_masks, segments, masks
