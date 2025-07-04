import wandb
import numpy as np
import pandas as pd
import egg_class_functions as ecf
import tensorflow as tf
from tf.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical # type: ignore
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

seg_data_path = "Data/processed/predicted_segmentation_data.csv"

df_pred = ecf.segmented_image_import(seg_data_path)
single_egg_df = df_pred.loc[df_pred["single"] == 1].reset_index(drop=True)
single_egg_df = single_egg_df.dropna()
single_egg_df["segment"] = single_egg_df.apply(ecf.rotate_and_pad_rgb_segment, axis=1)
single_egg_df['species'] = single_egg_df['species'].replace("aegypti_old", "aegypti")
single_egg_df['species'] = single_egg_df['species'].replace("albopictus_old", "albopictus")
train, test = train_test_split(single_egg_df, test_size=0.1)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.0277),
  layers.RandomBrightness(factor=0.2),
  layers.RandomContrast(factor=0.2)
])

X_train = train['segment']
X_train = np.stack(X_train.to_list()).astype(np.float32)
y_train = train['species']
X_test = test['segment']
X_test = np.stack(X_test.to_list()).astype(np.float32)
y_test = test['species']
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_onehot = to_categorical(y_train_encoded)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train_onehot, test_size=0.1, random_state=42)

train_ds = ecf.prepare_dataset(X_train_split, y_train_split, data_augmentation, 32)
val_ds = ecf.prepare_dataset(X_val_split, y_val_split, data_augmentation, 32, shuffle=False)

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
    "epochs": 10,
    "batch_size": 32,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "num_classes": 2
})


local_checkpoint = ModelCheckpoint(
    filepath="models/model.{epoch:02d}.h5",
    save_best_only=True,
    save_weights_only=False
)

wandb_checkpoint = WandbModelCheckpoint(
    filepath="models-wandb/model-{epoch:02d}.keras"
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs=10,
                    validation_data=val_ds,
                    callbacks=[WandbMetricsLogger(),
                               wandb_checkpoint,
                               local_checkpoint
                                ]
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
