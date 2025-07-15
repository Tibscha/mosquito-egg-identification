import os
import skimage
import egg_class_functions as ecf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import transform, filters, io
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, erosion, dilation, footprint_rectangle, remove_small_holes
from skimage.transform import rotate, resize
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input # type: ignore


def segmentation(image, model):
    """
    Perform image segmentation using the specified model.

    The function optionally reorders image axes if the input is in (H, W, C) format,
    then passes the image to the provided segmentation model.

    Args:
        image (numpy.ndarray): Input image, expected in shape (H, W, C) with 3 channels.
        model (object): A segmentation model with a `.segment()` method. For example,
                        a napari-convpaint model.

    Returns:
        numpy.ndarray: The segmentation result as returned by the model.
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


def resize_with_padding(image, target_width, target_height):
    """
    Resize an image while preserving its aspect ratio, and pad it to match the target dimensions.

    The function scales the input image to fit within the target size, maintaining the original 
    aspect ratio. The remaining space is filled with padding (black pixels). Additionally, a 
    boolean mask is returned that indicates the area occupied by the resized image 
    (i.e., the non-padded region).

    Args:
        image (ndarray): Input image as a NumPy array.
        target_width (int, optional): Target width in pixels. Default is 3088.
        target_height (int, optional): Target height in pixels. Default is 2076.

    Returns:
        tuple:
            image_padded (ndarray): The resized and padded image.
            mask (ndarray of bool): A boolean mask where `True` indicates the region
                                    corresponding to the original (resized) image.
    """
    original_height, original_width = image.shape[:2]
    target_aspect = target_width / target_height
    original_aspect = original_width / original_height

    if original_aspect > target_aspect:
        new_width = int(target_width)
        new_height = int(target_width / original_aspect)
    else:
        new_height = int(target_height)
        new_width = int(target_height * original_aspect)

    image_resized = transform.resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True)
    image_resized = image_resized.astype(np.uint8)

    pad_height = int(target_height - new_height)
    pad_width = int(target_width - new_width)

    pad_top = int(pad_height // 2)
    pad_bottom = int(pad_height - pad_top)
    pad_left = int(pad_width // 2)
    pad_right = int(pad_width - pad_left)

    if image_resized.ndim == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    image_padded = np.pad(image_resized, padding, mode='constant', constant_values=0)

    mask = np.zeros((int(target_height), int(target_width)), dtype=bool)
    mask[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = True

    return image_padded, mask


def egg_image_data_import(image_paths):
    """
    Imports images from a list of file paths and returns both the loaded images and 
    a DataFrame containing metadata for each image.

    The expected directory structure is:
        root/
            device_1/
                species_a/
                    image1.jpg
                    image2.jpg
                species_b/
            device_2/
                species_c/
            ...

    Each image must reside in a species subfolder inside a device folder.

    Args:
        image_paths (list of str): List of file paths to the images.

    Returns:
        list: List of loaded images as NumPy arrays.
        pandas.DataFrame: DataFrame with columns: 
                          - 'image_path': full image path,
                          - 'device': device folder name,
                          - 'species': species folder name.
    """
    data = []
    images = []
    for path in image_paths:
        try:
            image = io.imread(path)
            if image.ndim < 2:
                print(f"Skipping non-image file (too few dimensions): {path}")
                continue
        except Exception as e:
            print(f"Skipping invalid image file: {path} ({e})")
            continue

        species = os.path.basename(os.path.dirname(path))
        device = os.path.basename(os.path.dirname(os.path.dirname(path)))
        name = os.path.basename(path)
        data.append({
            'image_path': path,
            'device': device,
            'species': species,
            'image': image,
            'name': name
        })
        images.append(image)

    df = pd.DataFrame(data)
    if images == []:
        return print("No images in image path variable")
    return images, df


def batch_image_import(image_paths):
    """
    Imports multiple images from a list of file paths.

    Any files that cannot be read as valid images will be skipped.

    Args:
        image_paths (list of str): List of file paths to image files.

    Returns:
        list: List of successfully loaded images as NumPy arrays.
    """
    images = []
    for path in image_paths:
        try:
            image = io.imread(path)
            if image.ndim < 2:
                print(f"Skipping non-image file (too few dimensions): {path}")
                continue
            else:
                images.append(image)
        except Exception as e:
            print(f"Skipping invalid image file: {path} ({e})")
            continue
    if images == []:
        return print("No images in image path variable")
    return images


def segmented_image_import(data_path):
    """
    Imports the segmented csv data then importing
    multiple segmented images and masks from paths inside the csv.

    Args:
        data_path (str): Data path to the segmented csv data.

    Returns:
        Pandas.DataFrame: DataFrame with succesfully loaded images and masks.
    """
    df = pd.read_csv(data_path)
    segments = []
    masks = []
    df["segment"] = pd.Series([None] * len(df))
    df["mask"] = pd.Series([None] * len(df))
    for i, row in df.iterrows():
        try:
            #df.at[i, "segment"] = io.imread(row.segment_path)
            #df.at[i, "mask"] = io.imread(row.segment_mask_path)
            segments.append(io.imread(row.segment_path))
            masks.append(io.imread(row.segment_mask_path))
        except Exception as e:
            print(f"Skipping invalid image file: {row.segment_path} ({e})")
            continue
    df["segment"] = segments
    df["mask"] = masks
    return df


def region_separation(segment_mask):
    """
    Refines a segmentation mask and extracts distinct labeled regions.

    The function performs the following steps:
        - Removes small holes in regions labeled as 2
        - Applies morphological erosion to eliminate small connections or noise
        - Removes small objects below a size threshold
        - Applies dilation to restore approximate original shapes
        - Labels connected regions

    Args:
        segment_mask (ndarray): The input segmentation mask (e.g. with label 2 indicating the foreground).

    Returns:
        tuple:
            regions (skimage.measure._regionprops.RegionProperties): 
                List of region properties for each labeled region.
            labeled_overlay (ndarray): Labeled image where each region has a unique integer ID.
    """
    mask_cleaned = remove_small_holes(segment_mask == 2, area_threshold=5000)
    mask_cleaned = erosion(mask_cleaned, footprint=footprint_rectangle((25, 25)))
    mask_cleaned = remove_small_objects(mask_cleaned, min_size=20000)
    mask_cleaned = dilation(mask_cleaned, footprint_rectangle((25, 25)), mode='ignore')
    labeled_overlay = label(mask_cleaned)
    regions = regionprops(labeled_overlay)

    return regions, labeled_overlay


def region_processing(image, labeled_overlay, region):
    """
    Extracts and processes a single labeled region from an image.

    The function crops the specified region from the original image, creates a mask, 
    and calculates various shape and sharpness features. It returns the masked region, 
    the corresponding binary mask, and a dictionary of region properties and features.

    Args:
        image (ndarray): The original RGB image.
        labeled_overlay (ndarray): Labeled image where each region has a unique integer ID.
        region (skimage.measure._regionprops.RegionProperties): Region to process.

    Returns:
        tuple:
            masked_image (ndarray): The RGB image of the extracted region, masked with the region shape.
            mask (ndarray): Binary mask of the region within the bounding box.
            data (dict): Dictionary of region properties and computed features including:
                         - angle, area, perimeter, roundness, axis lengths,
                           length/width ratio and sharpness measures.
    """
    
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
    
    angle_deg = -np.degrees(angle)
    masked_image = rotate(masked_image, angle_deg, resize=True, preserve_range=False)
    mask = rotate(mask, angle_deg, resize=True, preserve_range=False)

    data = {
        'area': area,
        'perimeter': perimeter,
        'roundness': roundness,
        'length': length,
        'width': width,
        'len_wid_ratio': ratio,
        'laplacian' : laplacian,
        'edge': edge,
    }

    return masked_image, mask, data


def make_key(row):
    return f"image_{row['image_index']}_segment_{row['region_index']}.png"


def show_images(image_df):
    """
    Displays the first 25 images from a DataFrame in a 5x5 grid.

    Assumes that:
        - The DataFrame contains a 'segment' column with image arrays.
        - The DataFrame contains 'image_index' and 'region_index' columns 
          used for titling the subplots.

    Args:
        image_df (pandas.DataFrame): DataFrame containing image data 
                                     and metadata for display.

    Returns:
        None
    """
    fig, axes = plt.subplots(5,5,figsize=(20,12))
    ax = axes.flatten()
    for i in range(25):
        ax[i].set_title(f"image_{image_df.loc[i, "image_index"]}_segment_{image_df.loc[i, "region_index"]}")
        ax[i].imshow(image_df.loc[i, "segment"])
        ax[i].axis("off")
    plt.tight_layout()


def pad_rgb_segment(row, output_shape=(200, 100)):
    """
    Crops and resizes both the RGB image and its mask using the mask,
    preserving aspect ratio and padding to the output shape.

    Returns:
        dict: {
            'segment': padded RGB image,
            'mask': padded binary mask
        }
    """
    image = row["segment"]
    mask = row["mask"]
    mask = mask / 255
    mask = mask.astype(np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        return {
            "segment": image,
            "mask": mask
        }

    coords = np.argwhere(mask)
    if coords.size == 0:
        empty_rgb = np.zeros((*output_shape, 3), dtype=np.uint8)
        empty_mask = np.zeros(output_shape, dtype=np.uint8)
        return {
            "segment": empty_rgb,
            "mask": empty_mask
        }

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped_image = image[y0:y1, x0:x1, :]
    cropped_mask = mask[y0:y1, x0:x1]

    obj_h, obj_w = cropped_image.shape[:2]
    target_h, target_w = output_shape
    scale = min(target_h / obj_h, target_w / obj_w)
    new_h = int(obj_h * scale)
    new_w = int(obj_w * scale)

    resized_image = resize(cropped_image, (new_h, new_w, 3), preserve_range=True, anti_aliasing=True)
    resized_mask = resize(cropped_mask, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False)

    padded_image = np.zeros((target_h, target_w, 3), dtype=resized_image.dtype)
    padded_mask = np.zeros((target_h, target_w), dtype=resized_mask.dtype)

    start_y = (target_h - new_h) // 2
    start_x = (target_w - new_w) // 2
    padded_image[start_y:start_y+new_h, start_x:start_x+new_w, :] = resized_image
    padded_mask[start_y:start_y+new_h, start_x:start_x+new_w] = resized_mask

    padded_image = np.clip(padded_image, 0, 255).astype(np.uint8)
    padded_mask = (padded_mask > 0.5).astype(np.uint8)

    return {
        "segment": padded_image,
        "mask": padded_mask
    }


def prepare_dataset_tf(X, y, data_augmentation=None, batch_size=32, shuffle=True):
    """
    Erstellt ein tf.data.Dataset aus Bildern und Labels und bereitet es korrekt für EfficientNetV2B0 vor.
    
    - X: numpy array oder list of arrays, shape (num_samples, 200, 100, 3)
    - y: list oder array von int64-Labels
    - batch_size: Größe pro Batch
    - shuffle: True/False – ob das Dataset geshuffelt wird
    """

    def preprocess(img, label):
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        label = tf.cast(label, tf.int32)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if data_augmentation:
        ds = ds.map(lambda x, y: (data_augmentation(x, training = True), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def prepare_dataset_alb(X, y, augment_fn=None, batch_size=32, shuffle=True):
    """
    Erstellt ein tf.data.Dataset aus Bildern und Labels und bereitet es korrekt für EfficientNetV2B0 vor.
    
    - X: numpy array oder list of arrays, shape (num_samples, 200, 100, 3)
    - y: list oder array von int64-Labels
    - batch_size: Größe pro Batch
    - shuffle: True/False ob das Dataset geshuffelt wird
    """

    def preprocess_train(img, mask, label):
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        label = tf.cast(label, tf.int32)
        return img, mask, label
    
    def preprocess_val(img, label):
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        label = tf.cast(label, tf.int32)
        return img, label
    
    images = np.stack(X['segment'].values)
    mask = np.stack(X['mask'].values)
    
    if augment_fn:
        ds = tf.data.Dataset.from_tensor_slices((images, mask, y))
        ds = ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda img, mask, label: (img, label), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((images, y))
        ds = ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
        
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds
