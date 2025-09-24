import os
import glob
import wandb
import re
import gc
import numpy as np
import pandas as pd
import egg_class_functions as ecf
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras import layers # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow import keras
from tensorflow.data import AUTOTUNE # type: ignore

seg_data_path = "Data/processed/predicted_segmentation_data.csv"
image_paths = sorted(glob.glob("Data/raw/*/**/*.*", recursive=True), key=lambda x: (os.path.dirname(x), os.path.basename(x)))
metrics_csv = "Data/processed/model_metrics.csv"
BATCH_SIZE = 32
EPOCHS = 20
SEEDS = [2, 4, 6, 8, 10]
FINE_TUNE_EPOCHS = 20
height = 200
width = 100

all_steps = EPOCHS * 38
warmup_steps = all_steps * 0.15
decay_steps = all_steps * 0.85 
all_steps_ft = FINE_TUNE_EPOCHS * 38
warmup_steps_ft = all_steps_ft * 0.15
decay_steps_ft = all_steps_ft * 0.85 

run = "EfficientNetV2B0_20_epochs"
optimizer_name = "AdamW"
layer_names = "flatten_dense"


df_pred = ecf.segmented_image_import(seg_data_path)

for i, row in df_pred.iterrows():
    if row["device"] == "microscope":
        m = re.match(r"(ag|ap)_(\d+)", str(row["name"]))
        if m:
            num = int(m.group(2))
            if 1 <= num <= 200:
                df_pred.at[i, "age"] = "prior"
            elif 201 <= num <= 400:
                df_pred.at[i, "age"] = "fresh"
            elif 401 <= num <= 600:
                df_pred.at[i, "age"] = "dried"
    elif row["device"] == "phone":
        date = str(row["name"])[:10] 
        if date in ["2025-08-06", "2025-08-07"]:
            df_pred.at[i, "age"] = "fresh"
        elif date == "2025-08-12":
            df_pred.at[i, "age"] = "dried"


df_pred = df_pred.dropna()
results = df_pred.apply(ecf.pad_rgb_segment, output_shape=(height, width), axis=1)
df_pred["segment"] = results.apply(lambda x: x["segment"])
df_pred["mask"] = results.apply(lambda x: x["mask"])
df_pred['species'] = df_pred['species'].replace("aegypti_old", "aegypti")
df_pred['species'] = df_pred['species'].replace("albopictus_old", "albopictus")

color_trans = A.Compose([
# Image Capture Variance
A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=.5),
A.PlanckianJitter(p=.5),
#A.ImageCompression(quality_lower=80, quality_upper=100, p=.25),
A.Defocus(radius=(1, 3), p=.25),
A.RandomGamma(gamma_limit=(80, 120), p=.25),
A.MotionBlur(blur_limit=(3, 3), p=.25),
#A.Downscale(scale_min=0.8, scale_max=1, p=.25),
# Color Changes
A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=.5),
A.ChannelDropout(channel_drop_range=(1, 1), p=.25),
A.RandomShadow(shadow_roi=(0.3,0,0.7,1),p=0.25),
# Noise
A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=.25),
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

X_all = df_pred.loc[:, ['segment', 'mask']]
y_all = df_pred['species']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_all)
y_onehot = to_categorical(y_encoded)

# globaler Testsplit (bleibt für alle Subsets gleich)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

subsets = {
    "all": lambda df: df,
    "microscope": lambda df: df[(df['device'] == 'microscope') & (df['age'] != 'prior')],
    "phone": lambda df: df[df['device'] == 'phone'],
    "fresh": lambda df: df[(df['age'] == 'fresh') & (df['age'] != 'prior')],
    "dried": lambda df: df[(df['age'] == 'dried') & (df['age'] != 'prior')],
    "comparison": lambda df: df[df['age'].isin(['fresh', 'dried'])]
}

subset_train_splits = {}

subset_train_splits = {}

for subset_name, filter_func in subsets.items():
    df_sub = filter_func(df_pred.loc[X_train.index])
    
    if len(df_sub) == 0:
        continue

    X_subset = df_sub.loc[:, ['segment', 'mask']]
    y_encode_sub = y_encoded[df_sub.index]
    y_subset = y_onehot[df_sub.index]

    for seed in SEEDS:
        X_training, X_val, y_train, y_val = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=seed, stratify=y_encode_sub
        )
        
        model_name = f"ENet_{subset_name}_seed_{seed}"

        train_ds = ecf.prepare_dataset_alb(X_training, y_train, tf_albumentations_augment, BATCH_SIZE, repeat=True)
        val_ds = ecf.prepare_dataset_alb(X_val, y_val, None, BATCH_SIZE, shuffle=False, repeat=True)
        
        base_model = tf.keras.applications.EfficientNetV2B0(
            input_shape=(height, width, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            layers.Flatten(),
            #layers.GlobalAveragePooling2D(),
            #layers.GlobalMaxPooling2D(),
            #layers.Dropout(rate=0.25),
            layers.Dense(128, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])


        wandb.init(project="egg-classification", 
            name=model_name,
            config={
            "architecture": "EfficientNetV2B0",
            "input_shape": (height, width, 3),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "adamw",
            "loss": "categorical_crossentropy",
            "num_classes": 2
        })

        cosinedecay = keras.optimizers.schedules.CosineDecay(
            decay_steps = decay_steps,
            initial_learning_rate = 0.0,
            warmup_steps = warmup_steps,
            warmup_target = 0.0001,
        )

        optimizer = keras.optimizers.AdamW(learning_rate=cosinedecay)

        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    )
        
        history = model.fit(train_ds,
                            epochs=EPOCHS,
                            validation_data=val_ds,
                            steps_per_epoch=38,
                            validation_steps=9,
                            callbacks=[WandbMetricsLogger(),                            
                                        ],
                            )
        
        #model.save("tmp_model.keras")
        #del model

        #model = keras.models.load_model("tmp_model.h5")
        
        fine_tune_at = int(len(base_model.layers) * 0.8)

        for i, layer in enumerate(base_model.layers):
            if i >= fine_tune_at:
                layer.trainable = True
            else:
                layer.trainable = False

        cosinedecay = keras.optimizers.schedules.CosineDecay(
            decay_steps = decay_steps_ft,
            initial_learning_rate = 00.,
            warmup_steps = warmup_steps_ft,
            warmup_target = 0.0001,
        )

        optimizer = keras.optimizers.AdamW(learning_rate=cosinedecay)
        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    )


        history = model.fit(train_ds,
                            epochs=FINE_TUNE_EPOCHS,
                            validation_data=val_ds,
                            steps_per_epoch=38,
                            validation_steps=9,
                            callbacks=[WandbMetricsLogger(),                           
                                        ],
                            )

        for eval_name, filter_func in subsets.items():
            df_eval = filter_func(df_pred.loc[X_test.index])  # nur globales Test-Set
            if len(df_eval) == 0:
                continue
            
            X_subset_test = df_eval.loc[:, ['segment', 'mask']]
            y_subset_test = y_onehot[df_eval.index]  # Labels aus globalem Test-Set
            
            # Dataset für die Evaluation vorbereiten
            test_ds = ecf.prepare_dataset_alb(X_subset_test, y_subset_test, None, BATCH_SIZE, shuffle=False)

            y_test_true = []
            y_test_pred = []

            for x, y in test_ds:
                preds = model.predict(x)
                y_test_true.extend(np.argmax(y.numpy(), axis=1))
                y_test_pred.extend(np.argmax(preds, axis=1))

            metrics = {
                "run": run,
                "subset_name": subset_name,
                "eval_subset": eval_name,
                "seed": seed,
                "optimizer": optimizer_name,
                "layers": layer_names,
                "accuracy_test": accuracy_score(y_test_true, y_test_pred),
                "precision_test": precision_score(y_test_true, y_test_pred, average='macro'),
                "recall_test": recall_score(y_test_true, y_test_pred, average='macro'),
                "f1_test": f1_score(y_test_true, y_test_pred, average='macro'),
                "precision_test_w": precision_score(y_test_true, y_test_pred, average='weighted'),
                "recall_test_w": recall_score(y_test_true, y_test_pred, average='weighted'),
                "f1_test_w": f1_score(y_test_true, y_test_pred, average='weighted'),
                "confusion_matrix": confusion_matrix(y_test_true, y_test_pred)
            }

            df_metrics = pd.DataFrame([metrics])
            if os.path.exists(metrics_csv):
                df_metrics.to_csv(metrics_csv, mode='a', header=False, index=False)
            else:
                df_metrics.to_csv(metrics_csv, index=False)

        wandb.finish()
        del model, history, metrics, train_ds, val_ds
        tf.keras.backend.clear_session()
        gc.collect()
