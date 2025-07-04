import glob
import os
import skimage
import numpy as np
import pandas as pd
import egg_class_functions as ecf
from joblib import load
from napari_convpaint.conv_paint_model import ConvpaintModel
from skimage.util import img_as_ubyte

seg_model = ConvpaintModel(model_path="models/Enet_v7.pkl")
seg_class_model = load("models/seg_ident_classifier_v3.joblib")
seg_data_path = "Data/processed/predicted_segmentation_data.csv"
image_paths = sorted(glob.glob("Data/raw/microscope/**/*.*", recursive=True), key=lambda x: (os.path.dirname(x), os.path.basename(x)))
height = 2076
width = 3088
features = ('area', 'perimeter', 'roundness', 'length', 'width', 'len_wid_ratio', 'laplacian', 'edge')

images, df = ecf.egg_image_data_import(image_paths)
df['name'] = df['name'].str.replace(r'\.[^.]+$', '', regex=True)
df['content_mask'] = pd.Series()

resize_mask = []
for i, row in df.iterrows():
    if row['image'].shape[:2] != (height, width):
        image, content_mask = ecf.resize_with_padding(row['image'], width, height)
    else:
        image = row['image']
        content_mask = np.ones((height, width), dtype=bool)
    df.at[i, 'image'] = image
    df.at[i, 'content_mask'] = content_mask

data = []
masked_images = []
masks = []
for i, row in df.iterrows():
    segment_mask = ecf.segmentation(row['image'], seg_model)
    if row['image'].shape[:2] != (height, width):
        content_mask = row['content_mask']
        segment_mask[~content_mask] = 1
    regions, labeled_overlay = ecf.region_separation(segment_mask)
    for j, region in enumerate(regions):
        masked_image, mask, parameters = ecf.region_processing(row['image'], labeled_overlay, region)
        segment_path = f"Data/processed/segment_images/image_{i}_segment_{j}.png"
        segment_mask_path = f"Data/processed/segment_masks/image_{i}_mask_{j}.png"
        df_info = df.iloc[i].to_dict()
        combined = {**df_info, 'segment_path': segment_path, 'segment_mask_path': segment_mask_path, 'image_index': i, 'region_index': j, **parameters}
        data.append(combined)
        masked_images.append(masked_image)
        masks.append(mask)
        skimage.io.imsave(segment_path, img_as_ubyte(masked_image))
        skimage.io.imsave(segment_mask_path, img_as_ubyte(mask))
data_df = pd.DataFrame(data)
data_df.to_csv("Data/processed/segmentation_data.csv", index=False)

X = data_df.loc[:, features]
y_pred = seg_class_model.predict(X)
class_names = ['cut-off', 'multi', 'single']
y_pred_df = pd.DataFrame(y_pred, columns=class_names)
y_pred_df.index = data_df.index
df_pred = pd.concat([data_df, y_pred_df], axis=1)
df_pred.to_csv(seg_data_path, index=False)
