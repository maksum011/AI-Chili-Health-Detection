import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os

# ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_PATH = "model_cabai.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("="*50)
print("Model berhasil dimuat")
print("="*50)

# ==========================================================
# LOKASI GAMBAR
# ==========================================================

img_path = "dataset/sehat/sehat1.png"

# Contoh:
# img_path = "dataset/tidak_sehat/rusak1.png"

# ==========================================================
# CEK FILE
# ==========================================================

if not os.path.exists(img_path):

    print("Gambar tidak ditemukan!")

    exit()

# ==========================================================
# LOAD GAMBAR
# ==========================================================

img = image.load_img(

    img_path,

    target_size=(224,224)

)

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = preprocess_input(img_array)

# ==========================================================
# PREDIKSI
# ==========================================================

prediction = model.predict(img_array, verbose=0)

confidence = prediction[0][0]

# ==========================================================
# HASIL
# ==========================================================

if confidence < 0.5:

    hasil = "Cabai Sehat"

    persen = (1-confidence)*100

else:

    hasil = "Cabai Tidak Sehat"

    persen = confidence*100

# ==========================================================
# TAMPILKAN GAMBAR
# ==========================================================

gambar = Image.open(img_path)

plt.figure(figsize=(6,6))

plt.imshow(gambar)

plt.axis("off")

plt.title(

    hasil,

    fontsize=18,

    color="green" if hasil=="Cabai Sehat" else "red"

)

plt.show()

# ==========================================================
# HASIL
# ==========================================================

print("\n")

print("="*50)

print("HASIL PREDIKSI")

print("="*50)

print("Nama File :", os.path.basename(img_path))

print("Prediksi  :", hasil)

print("Confidence: {:.2f}%".format(persen))

print("="*50)