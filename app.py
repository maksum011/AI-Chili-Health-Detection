import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="AI Chili Health Detection",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_cabai.keras")

model = load_model()

# =====================================================
# CLASS
# =====================================================

CLASS_NAMES = [
    "Cabai Sehat",
    "Cabai Tidak Sehat"
]

# =====================================================
# CSS MODERN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
    background:#061326;
    color:white;
}

/* Background */

.stApp{
    background:linear-gradient(
    135deg,
    #061326,
    #0b1f3a,
    #10284d);
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#081729;
    border-right:1px solid #1b4d72;
}

/* Card */

.card{

    background:rgba(16,32,60,.90);

    border:1px solid rgba(0,212,255,.25);

    border-radius:22px;

    padding:28px;

    margin-bottom:20px;

    box-shadow:0 0 25px rgba(0,212,255,.15);

}

/* Glass */

.glass{

    background:rgba(255,255,255,.04);

    backdrop-filter:blur(18px);

    border:1px solid rgba(255,255,255,.08);

    border-radius:20px;

    padding:22px;

}

/* Judul */

.big-title{

    font-size:48px;

    font-weight:800;

    color:white;

}

.sub-title{

    font-size:20px;

    color:#00d9ff;

}

/* Result Card */

.result-card{

    background:#0f2345;

    border-radius:20px;

    padding:25px;

    border:2px solid #00d9ff;

    box-shadow:0 0 20px rgba(0,212,255,.30);

}

/* Metric */

.metric{

    background:#13294f;

    border-radius:18px;

    padding:18px;

    text-align:center;

    border:1px solid #1e4975;

}

/* Text */

h1,h2,h3,h4,h5,h6{
    color:white;
}

p,label,span,div{
    color:#ECECEC;
}

/* Button */

.stButton>button{

    width:100%;

    background:linear-gradient(90deg,#00c6ff,#0072ff);

    color:white;

    border:none;

    border-radius:12px;

    font-weight:700;

    height:52px;

    transition:.3s;

}

.stButton>button:hover{

    transform:scale(1.03);

    box-shadow:0 0 15px cyan;

}

/* Upload */

[data-testid="stFileUploader"]{

    border:2px dashed #00d9ff;

    border-radius:15px;

    background:#0d203d;

    padding:10px;

}

/* Footer */

footer{

    visibility:hidden;

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 🌶️ AI CHILI")

    st.markdown("### Health Detection")

    st.markdown("---")

    st.success("🟢 Model Loaded")

    st.markdown("## 🤖 Model")

    st.info("""

MobileNetV2

TensorFlow

Python

224 x 224 Pixel

2 Class

""")

    st.markdown("---")

    st.markdown("## 📊 Dataset")

    st.write("🟢 Cabai Sehat")

    st.write("🔴 Cabai Tidak Sehat")

    st.markdown("---")

    st.markdown("## 👨‍💻 Developer")

    st.write("teknik informatika")

    st.write("Universitas Al Asyariah Mandar")

    st.markdown("---")

    st.caption("AI Chili Health Detection v1.0")

    # =====================================================
# HEADER
# =====================================================

st.markdown("""

<div class="card">

<div class="big-title">
🌶️ AI DETEKSI KESEHATAN CABAI
</div>

<div class="sub-title">
Artificial Intelligence • Computer Vision • Deep Learning
</div>

<br>

Sistem ini menggunakan model <b>Convolutional Neural Network (CNN)</b>
berbasis <b>MobileNetV2</b> untuk mengidentifikasi kondisi kesehatan
buah cabai secara otomatis dari gambar yang diunggah.

</div>

""", unsafe_allow_html=True)

# =====================================================
# METRIC
# =====================================================

col1,col2,col3,col4 = st.columns(4)

with col1:

    st.markdown("""

    <div class="metric">

    <h2>2</h2>

    <p>Kelas</p>

    </div>

    """,unsafe_allow_html=True)

with col2:

    st.markdown("""

    <div class="metric">

    <h2>CNN</h2>

    <p>Model AI</p>

    </div>

    """,unsafe_allow_html=True)

with col3:

    st.markdown("""

    <div class="metric">

    <h2>224×224</h2>

    <p>Input Size</p>

    </div>

    """,unsafe_allow_html=True)

with col4:

    st.markdown("""

    <div class="metric">

    <h2>TensorFlow</h2>

    <p>Framework</p>

    </div>

    """,unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

# =====================================================
# UPLOAD
# =====================================================

uploaded_file = st.file_uploader(

    "📤 Upload Gambar Cabai",

    type=["jpg","jpeg","png"]

)

# =====================================================
# JIKA BELUM ADA GAMBAR
# =====================================================

if uploaded_file is None:

    st.info("Silakan upload gambar cabai untuk memulai proses deteksi.")

    st.stop()

# =====================================================
# LOAD IMAGE
# =====================================================

image_pil = Image.open(uploaded_file).convert("RGB")

left,right = st.columns([1.1,1])

# =====================================================
# PREVIEW
# =====================================================

with left:

    st.markdown("""

    <div class="glass">

    <h3>🖼 Preview Gambar</h3>

    </div>

    """,unsafe_allow_html=True)

    st.image(

        image_pil,

        use_container_width=True

    )

    st.write("**Nama File** :",uploaded_file.name)

    st.write(

        "**Resolusi** :",

        f"{image_pil.size[0]} x {image_pil.size[1]}"

    )

# =====================================================
# ANALISIS AI
# =====================================================

with right:

    st.markdown("""

    <div class="glass">

    <h3>🤖 Analisis AI</h3>

    </div>

    """,unsafe_allow_html=True)

    progress = st.progress(0)

    status = st.empty()

    for i in range(100):

        progress.progress(i+1)

        status.write(

            f"Memproses gambar... {i+1}%"

        )

        time.sleep(0.01)

    status.success("Analisis selesai.")

    # =====================================================
# PREPROCESS IMAGE
# =====================================================

img = image_pil.resize((224,224))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = preprocess_input(img_array)

# =====================================================
# PREDIKSI MODEL
# =====================================================

prediction = model.predict(

    img_array,

    verbose=0

)

confidence = prediction[0][0]

# =====================================================
# HASIL PREDIKSI
# =====================================================

if confidence < 0.5:

    hasil = "Cabai Sehat"

    persen = (1-confidence)*100

    warna = "#00E676"

    icon = "✅"

else:

    hasil = "Cabai Tidak Sehat"

    persen = confidence*100

    warna = "#FF5252"

    icon = "❌"

# =====================================================
# HASIL AI
# =====================================================

st.markdown("<br>",unsafe_allow_html=True)

st.markdown(f"""

<div class="result-card">

<h2>{icon} HASIL ANALISIS AI</h2>

<h1 style="color:{warna};font-size:55px;">
{hasil}
</h1>

<h2 style="color:white;">
Confidence
</h2>

<h1 style="color:{warna};">
{persen:.2f}%
</h1>

</div>

""",unsafe_allow_html=True)

# =====================================================
# CONFIDENCE BAR
# =====================================================

st.markdown("### 📊 Tingkat Keyakinan Model")

st.progress(float(persen/100))

# =====================================================
# TABEL HASIL
# =====================================================

st.markdown("## 📋 Detail Prediksi")

df = pd.DataFrame({

    "Parameter":[

        "Model",

        "Framework",

        "Ukuran Input",

        "Prediksi",

        "Confidence"

    ],

    "Hasil":[

        "MobileNetV2",

        "TensorFlow",

        "224 x 224",

        hasil,

        f"{persen:.2f}%"

    ]

})

st.dataframe(

    df,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# GRAFIK BAR
# =====================================================

st.markdown("## 📈 Confidence Visualization")

if hasil == "Cabai Sehat":

    nilai = [

        persen,

        100-persen

    ]

else:

    nilai = [

        100-persen,

        persen

    ]

fig, ax = plt.subplots(figsize=(8,4))

kelas = [

    "Cabai Sehat",

    "Cabai Tidak Sehat"

]

warna_bar = [

    "#00E676",

    "#FF5252"

]

bars = ax.bar(

    kelas,

    nilai,

    color=warna_bar,

    width=0.5

)

ax.set_ylim(0,100)

ax.set_ylabel("Confidence (%)")

ax.set_title("Probability")

ax.grid(

    axis="y",

    alpha=0.3

)

for bar in bars:

    tinggi = bar.get_height()

    ax.text(

        bar.get_x()+bar.get_width()/2,

        tinggi+2,

        f"{tinggi:.1f}%",

        ha="center",

        fontsize=11,

        fontweight="bold"

    )

st.pyplot(fig)