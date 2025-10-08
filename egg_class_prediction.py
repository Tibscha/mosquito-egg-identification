import os
import glob
import cv2
import numpy as np
import pandas as pd
import egg_class_functions as ecf
from joblib import load
from napari_convpaint.conv_paint_model import ConvpaintModel
from tensorflow import keras
from skimage import io
from skimage.util import img_as_ubyte
from skimage.draw import rectangle_perimeter

base_path = "Data/raw/microscope/albopictus_old"
model_name = "all"
class_model = keras.models.load_model(f"models/{model_name}.keras", compile=False)
seg_model = ConvpaintModel(model_path="models/Enet_v7.pkl")
seg_class_model = load("models/seg_ident_classifier_v3.joblib")
image_paths = sorted(
        glob.glob(
        f"{base_path}/*.*",
        recursive=True),
        key=lambda x: (os.path.dirname(x), os.path.basename(x)))
output_dir = f"{base_path}/annotated_images"
height_segment = 200
width_segment = 100
height = 2076
width = 3088
features = ('area', 'perimeter', 'roundness', 'length', 'width', 'len_wid_ratio', 'laplacian', 'edge')
seg_class_names = ['cut-off', 'multi', 'single']
class_names = ["aegypti", "albopictus"]

os.makedirs(output_dir, exist_ok=True)

images, df = ecf.egg_image_data_import(image_paths)
for i, row in df.iterrows():
    image_name = row["name"]
    image = row["image"]
    orig_h, orig_w = image.shape[:2]

    if image.shape[:2] != (height, width):
        image, content_mask = ecf.resize_with_padding(image, width, height)
    else:
        content_mask = np.ones((height, width), dtype=bool)

    segment_mask = ecf.segmentation(image, seg_model)
    if image.shape[:2] != (height, width):
        segment_mask[~content_mask] = 1

    regions, labeled_overlay = ecf.region_separation(segment_mask)

    annotated = image.copy()

    for j, region in enumerate(regions):
        masked_image, mask, parameters = ecf.region_processing(image, labeled_overlay, region)
        mask_uint8 = img_as_ubyte(mask)
        mask_image_uint8 = img_as_ubyte(masked_image)

        # Schritt 1: Pr�fen, ob es ein "single" Ei ist
        df_par = pd.DataFrame([parameters], columns=features)
        y_pred = seg_class_model.predict(df_par)
        pred_class = seg_class_names[np.argmax(y_pred)]

        # in dict konvertieren
        d = dict(enumerate(y_pred.flatten()))
        d['cut-off'] = d.pop(0)
        d['multi'] = d.pop(1)
        d['single'] = d.pop(2)

        if d["single"] != 1:  # nicht "single" \u2192 �berspringen
            continue

        # Schritt 2: padten und klassifizieren
        row_data = {"segment": mask_image_uint8, "mask": mask_uint8}
        padded = ecf.pad_rgb_segment(row_data, output_shape=(height_segment, width_segment))
        X = np.expand_dims(padded["segment"], axis=0)

        pred = class_model.predict(X)
        class_idx = np.argmax(pred)
        class_name = class_names[class_idx]
        confidence = float(pred[0][class_idx])
        confidence = round(confidence * 100, 2)
        annot_text = f"{class_name}, confidence: {confidence}%"
        # Schritt 3: Position im Originalbild annotieren
        minr, minc, maxr, maxc = region.bbox

        # Zeichne Rechteck
        rr, cc = rectangle_perimeter((minr, minc), end=(maxr, maxc), shape=annotated.shape)
        annotated[rr, cc] = [255, 0, 0]  # rotes Rechteck

        # Text hinzuf�gen (einfache Variante)
        from skimage.draw import rectangle
        from skimage import img_as_ubyte
        import cv2  # bessere Schrift

        annotated = img_as_ubyte(annotated)
        cv2.putText(
            annotated,
            annot_text,
            (minc, maxr + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    image_name = os.path.splitext(image_name)[0]
    save_path = os.path.join(output_dir, f"{image_name}_annotated.png")
    io.imsave(save_path, annotated)
    print(f"\u2705 Saved: {save_path}")
