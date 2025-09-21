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
image_paths = sorted(glob.glob("Data/raw/*/**/*.*", recursive=True), key=lambda x: (os.path.dirname(x), os.path.basename(x)))
height = 2076
width = 3088
features = ('area', 'perimeter', 'roundness', 'length', 'width', 'len_wid_ratio', 'laplacian', 'edge')
class_names = ['cut-off', 'multi', 'single']

images, df = ecf.egg_image_data_import(image_paths)
df['name'] = df['name'].str.replace(r'\.[^.]+$', '', regex=True)
df['content_mask'] = pd.Series()

resize_mask = []
data = []
for i, row in df.iterrows():
    if row['image'].shape[:2] != (height, width):
        image, content_mask = ecf.resize_with_padding(row['image'], width, height)
    else:
        image = row['image']
        content_mask = np.ones((height, width), dtype=bool)

    
    segment_mask = ecf.segmentation(image, seg_model)
    if image.shape[:2] != (height, width):
        segment_mask[~content_mask] = 1
    regions, labeled_overlay = ecf.region_separation(segment_mask)
    species = row['species']
    for j, region in enumerate(regions):
        masked_image, mask, parameters = ecf.region_processing(image, labeled_overlay, region)
        # predicting if its a single egg
        df_par = pd.DataFrame([parameters], columns=features)
        y_pred = seg_class_model.predict(df_par)
        pred_class = class_names[np.argmax(y_pred)]
        # saving the images
        segment_folder = f"Data/processed/segment_images/{species}/{pred_class}"
        segment_mask_folder = f"Data/processed/segment_masks/{species}/{pred_class}"
        os.makedirs(segment_folder, exist_ok = True)
        os.makedirs(segment_mask_folder, exist_ok = True)
        segment_path = f"Data/processed/segment_images/{species}/{pred_class}/image_{i}_segment_{j}.png"
        segment_mask_path = f"Data/processed/segment_masks/{species}/{pred_class}/image_{i}_mask_{j}.png"
        d = dict(enumerate(y_pred.flatten()))
        d['cut-off'] = d.pop(0)
        d['multi'] = d.pop(1)
        d['single'] = d.pop(2)
        df_info = df.drop(columns=["image"]).iloc[i].to_dict()
        combined = {**df_info,
                    'segment_path': segment_path,
                    'segment_mask_path': segment_mask_path,
                    'image_index': i,
                    'region_index': j,
                    **parameters,
                    **d
                    }
        data.append(combined)
        skimage.io.imsave(segment_path, img_as_ubyte(masked_image))
        skimage.io.imsave(segment_mask_path, img_as_ubyte(mask))

data_df = pd.DataFrame(data)
data_df.to_csv(seg_data_path, index=False)
