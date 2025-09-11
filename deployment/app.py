import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

# ------------------- إعدادات الصفحة -------------------
st.set_page_config(
    page_title="Disease Classification App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------- العنوان -------------------
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color:#2E86C1;">🩺 Image Classification Model</h1>
        <p style="font-size:18px; color:gray;">Upload an image and let the model predict the disease</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------- تحميل الموديل -------------------
model = load_model('https://drive.google.com/file/d/1nT2zuvx6jvkcmujiFQ7-ZMrsB_QnCU2w/view?usp=drive_link')

# أسماء الكلاسات
data_cat = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
img_height, img_width = 224, 224

# ------------------- رفع الصورة -------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    # تجهيز الصورة للموديل
    image = image.resize((img_height, img_width))
    img_arr = img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    # التنبؤ
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    # ------------------- عرض النتيجة -------------------
    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    st.markdown(
        f"""
        <div style="background-color:#F4F6F6; padding:20px; border-radius:15px; text-align:center; box-shadow: 2px 2px 8px #aaa;">
            <h2 style="color:#117A65;">Disease: <b>{predicted_class}</b></h2>
            <h3 style="color:#B03A2E;">Accuracy: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------- عرض نسب كل الكلاسات -------------------
    st.markdown("### 📊 Class Probabilities")
    for i, cat in enumerate(data_cat):
        st.write(f"**{cat}:** {score[i]*100:.2f}%")
        st.progress(float(score[i]))  # bar لكل class
        

else:
    st.info("👆 Please upload an image to start prediction.")
