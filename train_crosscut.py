# train_crosscut.py
import os, json, random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
SEED = 42

# Either use pre-split data/train & data/val folders…
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

# …or set this to True to auto-split from a single folder data/all/<class>/*
AUTO_SPLIT = False
ALL_DIR = DATA_DIR / "all"   # used only if AUTO_SPLIT=True
VAL_FRACTION = 0.2

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "crosscut_model.keras"   # recommended Keras format
LABELS_PATH = OUT_DIR / "class_indices.json"
HISTORY_PATH = OUT_DIR / "history.json"

# ----------------------------
# Utils
# ----------------------------
def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def count_images(root: Path):
    counts = {}
    total = 0
    for i in range(NUM_CLASSES):
        p = root / str(i)
        n = len([f for f in p.glob("*.*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if p.exists() else 0
        counts[str(i)] = n
        total += n
    return counts, total

def compute_class_weights(counts: dict):
    total = sum(counts.values())
    weights = {}
    for k, n in counts.items():
        weights[int(k)] = 0.0 if n == 0 else total / (NUM_CLASSES * n)
    return weights

# ----------------------------
# Datasets
# ----------------------------
def get_datasets():
    if AUTO_SPLIT:
        ds_train = tf.keras.utils.image_dataset_from_directory(
            directory=ALL_DIR,
            labels='inferred',
            label_mode='categorical',
            class_names=[str(i) for i in range(NUM_CLASSES)],
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            interpolation='bilinear',
            validation_split=VAL_FRACTION,
            subset='training',
            seed=SEED
        )
        ds_val = tf.keras.utils.image_dataset_from_directory(
            directory=ALL_DIR,
            labels='inferred',
            label_mode='categorical',
            class_names=[str(i) for i in range(NUM_CLASSES)],
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            interpolation='bilinear',
            validation_split=VAL_FRACTION,
            subset='validation',
            seed=SEED
        )
    else:
        ds_train = tf.keras.utils.image_dataset_from_directory(
            directory=TRAIN_DIR,
            labels='inferred',
            label_mode='categorical',
            class_names=[str(i) for i in range(NUM_CLASSES)],
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            interpolation='bilinear',
            seed=SEED
        )
        ds_val = tf.keras.utils.image_dataset_from_directory(
            directory=VAL_DIR,
            labels='inferred',
            label_mode='categorical',
            class_names=[str(i) for i in range(NUM_CLASSES)],
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            interpolation='bilinear',
            seed=SEED
        )

    # Scale to [0,1] to match your app (app also divides by 255.0)
    def to_float(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)  # [0,1]
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    ds_train = ds_train.map(to_float, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.map(to_float, num_parallel_calls=AUTOTUNE)

    ds_train = ds_train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds_train = ds_train.shuffle(1000).prefetch(AUTOTUNE)
    ds_val = ds_val.prefetch(AUTOTUNE)

    return ds_train, ds_val

# ----------------------------
# Model
# ----------------------------
def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    # No extra rescaling or preprocess_input; we train directly on [0,1].
    base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    base.trainable = False  # Stage 1: feature extractor

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model, base

# ----------------------------
# Train
# ----------------------------
def main():
    set_all_seeds()
    ds_train, ds_val = get_datasets()

    # Class weights for imbalance
    if AUTO_SPLIT:
        counts_train = {str(i): 0 for i in range(NUM_CLASSES)}
        for _, y in ds_train.unbatch():
            counts_train[str(int(tf.argmax(y).numpy()))] += 1
    else:
        counts_train, _ = count_images(TRAIN_DIR)

    class_weights = compute_class_weights(counts_train)
    print("Class counts:", counts_train)
    print("Class weights:", class_weights)

    model, base = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    hist1 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Fine-tune top layers
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    hist2 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Save label mapping/order
    class_indices = {i: str(i)}  # fixed order 0..5
    with open(LABELS_PATH, "w") as f:
        json.dump(class_indices, f, indent=2)

    # Save training history for later plotting
    history = {k: [*hist1.history.get(k, []), *hist2.history.get(k, [])]
               for k in set(list(hist1.history.keys()) + list(hist2.history.keys()))}
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved labels to: {LABELS_PATH}")
    print(f"Saved history to: {HISTORY_PATH}")

if __name__ == "__main__":
    main()
