# ==========================================================
# AI CHILI HEALTH DETECTION
# TRAIN.PY (2 CLASS)
# PART 1
# ==========================================================

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ==========================================================
# CEK TENSORFLOW
# ==========================================================

print("="*60)
print("TensorFlow :", tf.__version__)
print("="*60)

# ==========================================================
# KONFIGURASI
# ==========================================================

IMG_SIZE = (224,224)

BATCH_SIZE = 4

EPOCHS = 20

DATASET_PATH = "dataset"

MODEL_PATH = "model_cabai.keras"

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=0.20,

    rotation_range=15,

    zoom_range=0.15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    shear_range=0.10,

    horizontal_flip=True,

    brightness_range=[0.9,1.1],

    fill_mode="nearest"

)

valid_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=0.20

)

# ==========================================================
# TRAIN DATA
# ==========================================================

train_data = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training",

    shuffle=True

)

# ==========================================================
# VALIDATION DATA
# ==========================================================

val_data = valid_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation",

    shuffle=False

)

# ==========================================================
# INFORMASI DATASET
# ==========================================================

print("\n")

print("="*60)

print("Jumlah Training :", train_data.samples)

print("Jumlah Validation :", val_data.samples)

print("\nClass Index")

print(train_data.class_indices)

print("="*60)

# ==========================================================
# MEMBANGUN MODEL
# ==========================================================

print("\nMembangun Model MobileNetV2...")

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

)

# ==========================================================
# FREEZE LAYER
# ==========================================================

base_model.trainable = False

# ==========================================================
# CUSTOM HEAD
# ==========================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(

    128,

    activation="relu"

)(x)

x = Dropout(0.5)(x)

x = Dense(

    64,

    activation="relu"

)(x)

x = Dropout(0.3)(x)

output = Dense(

    1,

    activation="sigmoid"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=output

)

# ==========================================================
# COMPILE
# ==========================================================

optimizer = Adam(

    learning_rate=0.0001

)

model.compile(

    optimizer=optimizer,

    loss="binary_crossentropy",

    metrics=[

        "accuracy"

    ]

)

print("\n")

print("="*60)

print("MODEL BERHASIL DIBUAT")

print("="*60)

model.summary()

# ==========================================================
# CALLBACK
# ==========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-6,

    verbose=1

)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

callbacks = [

    early_stop,

    reduce_lr,

    checkpoint

]

# ==========================================================
# TRAINING
# ==========================================================

print("\n")
print("="*60)
print("MEMULAI TRAINING...")
print("="*60)

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)

# ==========================================================
# EVALUASI MODEL
# ==========================================================

print("\n")
print("="*60)
print("EVALUASI MODEL")
print("="*60)

loss, accuracy = model.evaluate(

    val_data,

    verbose=1

)

print(f"\nValidation Accuracy : {accuracy*100:.2f}%")
print(f"Validation Loss     : {loss:.4f}")

# ==========================================================
# LOAD MODEL TERBAIK
# ==========================================================

print("\nMemuat model terbaik...")

best_model = tf.keras.models.load_model(MODEL_PATH)

print("Model terbaik berhasil dimuat.")

# ==========================================================
# GRAFIK ACCURACY
# ==========================================================

plt.figure(figsize=(14,5))

# Accuracy
plt.subplot(1,2,1)

plt.plot(

    history.history["accuracy"],

    linewidth=3,

    color="blue",

    label="Training"

)

plt.plot(

    history.history["val_accuracy"],

    linewidth=3,

    color="green",

    label="Validation"

)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.legend()

# Loss
plt.subplot(1,2,2)

plt.plot(

    history.history["loss"],

    linewidth=3,

    color="red",

    label="Training"

)

plt.plot(

    history.history["val_loss"],

    linewidth=3,

    color="orange",

    label="Validation"

)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

# ==========================================================
# RINGKASAN
# ==========================================================

print("\n")
print("="*60)

print("TRAINING SELESAI")

print("="*60)

print("Model Disimpan :", MODEL_PATH)

print("\nLabel Dataset")

print(train_data.class_indices)

print("\nJumlah Training :", train_data.samples)

print("Jumlah Validation :", val_data.samples)

print(f"\nBest Validation Accuracy : {max(history.history['val_accuracy'])*100:.2f}%")

print(f"Best Validation Loss : {min(history.history['val_loss']):.4f}")

print("="*60)

print("Program selesai.")