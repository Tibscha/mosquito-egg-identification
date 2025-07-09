import wandb
import numpy as np
import egg_class_functions as ecf
import tensorflow as tf
import glob
import os
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical # type: ignore
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

seg_data_path = "Data/processed/predicted_segmentation_data.csv"
image_paths = sorted(glob.glob("Data/raw/microscope/**/*.*", recursive=True), key=lambda x: (os.path.dirname(x), os.path.basename(x)))
BATCH_SIZE = 32
EPOCHS = 100
AUTOTUNE = tf.data.AUTOTUNE

df_pred = ecf.segmented_image_import(seg_data_path)
single_egg_df = df_pred.loc[df_pred["single"] == 1].reset_index(drop=True)
single_egg_df = single_egg_df.dropna()
single_egg_df["segment"] = single_egg_df.apply(ecf.rotate_and_pad_rgb_segment, axis=1)
single_egg_df['species'] = single_egg_df['species'].replace("aegypti_old", "aegypti")
single_egg_df['species'] = single_egg_df['species'].replace("albopictus_old", "albopictus")
train, test = train_test_split(single_egg_df, test_size=0.1)

X_train = single_egg_df['segment']
X_train = np.stack(X_train.to_list()).astype(np.float32)
y_train = single_egg_df['species']
X_test = test['segment']
X_test = np.stack(X_test.to_list()).astype(np.float32)
y_test = test['species']
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_onehot = to_categorical(y_train_encoded)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train_onehot, test_size=0.2, random_state=42)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.0277),
  layers.RandomBrightness(factor=0.2),
  layers.RandomContrast(factor=0.2),
  layers.RandomColorJitter(
      value_range=(0, 1),
      brightness_factor=0.2,
      contrast_factor=0.2,
      saturation_factor=0.5,
      hue_factor=(0.5, 0.5)
      ),
  layers.RandomColorDegeneration(0.2),
  layers.RandomHue(factor=(0.5, 0.5), value_range=(0, 1)),
  layers.RandomSaturation(factor=0.5, value_range=(0, 1)),
  layers.RandomGaussianBlur(factor=0.2, sigma=(0.1, 0.4), value_range=(0, 1)),

])

train_ds = ecf.prepare_dataset_tf(X_train_split, y_train_split, data_augmentation, BATCH_SIZE)
val_ds = ecf.prepare_dataset_tf(X_val_split, y_val_split, data_augmentation, BATCH_SIZE, shuffle=False)

weight_for_albo = (1 / sum(single_egg_df['species'] == 'albopictus')) * (len(single_egg_df))
weight_for_aegy = (1 / sum(single_egg_df['species'] == 'aegypti')) * (len(single_egg_df))

class_weight = {0: weight_for_albo, 1: weight_for_aegy}

base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(200, 200, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

wandb.init(project="egg-classification", config={
    "architecture": "EfficientNetV2B0",
    "input_shape": (200, 200, 3),
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

optimizer = keras.optimizers.Adam(learning_rate=0.00001)

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
                    class_weight=class_weight
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