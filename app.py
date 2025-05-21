import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("C:\\Users\\admin\\Downloads\\model.h5")

# Set up the page
st.set_page_config(page_title="Mask Detection", layout="centered")

st.title("😷 Face Mask Detection")
st.write("Upload an image to check if the person is wearing a mask.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and preprocess the image
    image = Image.open(uploaded_file)
    img = np.array(image)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    input_img = np.reshape(img_normalized, [1, 128, 128, 3])

    # Make prediction
    pred = model.predict(input_img)
    pred_label = np.argmax(pred)

    # Display image and result side by side
    col1, col2 = st.columns([3,2])
    with col1:
        st.image(image, caption='Uploaded Image', width=400) # Smaller width

    with col2:
        if pred_label == 1:
            st.success("✅ The person is wearing a mask")
        else:
            st.error("❌ The person is not wearing a mask")
