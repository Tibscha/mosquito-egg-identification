import skimage
import os
import glob
import importlib
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import egg_class_functions as ecf
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow import keras
from tensorflow.data import AUTOTUNE # type: ignore

seg_data_path = "Data/processed/predicted_segmentation_data.csv"
image_paths = sorted(glob.glob("Data/raw/microscope/**/*.*", recursive=True), key=lambda x: (os.path.dirname(x), os.path.basename(x)))
BATCH_SIZE = 32
EPOCHS = 100
height = 200
width = 100

df_pred = ecf.segmented_image_import(seg_data_path)
single_egg_df = df_pred.loc[df_pred["single"] == 1].reset_index(drop=True)
single_egg_df = single_egg_df.dropna()
results = single_egg_df.apply(ecf.pad_rgb_segment, output_shape=(height, width), axis=1)
single_egg_df["segment"] = results.apply(lambda x: x["segment"])
single_egg_df["mask"] = results.apply(lambda x: x["mask"])
single_egg_df['species'] = single_egg_df['species'].replace("aegypti_old", "aegypti")
single_egg_df['species'] = single_egg_df['species'].replace("albopictus_old", "albopictus")

X_train = single_egg_df.loc[:, ['segment', 'mask']]
y_train = single_egg_df['species']

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_onehot = to_categorical(y_train_encoded)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train_onehot, test_size=0.2, random_state=42)

color_trans = A.Compose([
    # Image Capture Variance
    #A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=.5),
    #A.PlanckianJitter(p=.5),
    #A.ImageCompression(quality_lower=75, quality_upper=100, p=.25),
    #A.Defocus(radius=(1, 3), p=.25),
    #A.RandomGamma(gamma_limit=(80, 120), p=.25),
    #A.MotionBlur(blur_limit=(3, 3), p=.25),
    #A.Downscale(scale_min=0.75, scale_max=1, p=.25),
    # Color Changes
    #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=.5),
    #A.ChannelDropout(channel_drop_range=(1, 1), p=.25),
    #A.RandomShadow(shadow_roi=(0.3,0,0.7,1),p=0.25),
    # Noise
    #A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=.25),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=(0, 0.0625), scale_limit=0.0, rotate_limit=(-5, 5), p=0.5),
])


def albumentations_augment(image, mask):
    image = image.astype(np.uint8)
    augmented = color_trans(image=image)
    #augmented = augmented * mask

    aug_image = augmented['image'].astype(np.float32)
    aug_image *= mask[..., np.newaxis]

    return aug_image


def tf_albumentations_augment(image, mask, label):
    aug_image = tf.numpy_function(albumentations_augment, [image, mask], tf.float32)
    aug_image.set_shape(image.shape)
    return aug_image, mask, label

train_ds = ecf.prepare_dataset_alb(X_train_split, y_train_split, tf_albumentations_augment, BATCH_SIZE)
val_ds = ecf.prepare_dataset_alb(X_val_split, y_val_split, None, BATCH_SIZE, shuffle=False)

base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(height, width, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(rate=0.25),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

wandb.init(project="egg-classification", config={
    "architecture": "EfficientNetV2B0",
    "input_shape": (height, width, 3),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "num_classes": 2
})


local_checkpoint = ModelCheckpoint(
    filepath="models/model.{epoch:02d}.h5",
    save_best_only=True,
    save_weights_only=False
)

#wandb_checkpoint = WandbModelCheckpoint(
#    filepath="models-wandb/model-{epoch:02d}.keras",
#    save_best_only=True
#)

optimizer = keras.optimizers.Adam(learning_rate=0.000005)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              )

history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    callbacks=[WandbMetricsLogger(),
                               #wandb_checkpoint                               
                                ],
                    )

y_true = []
y_pred = []

for x, y in val_ds:
    preds = model.predict(x)
    y_true.extend(np.argmax(y.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_true,
    preds=y_pred,
    class_names=["aegypti", "albopictus"]
)})

wandb.finish()