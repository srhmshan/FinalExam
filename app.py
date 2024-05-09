import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('tsdr_model.h5')
    return model

model = load_model()

classes = {5: 'Speed limit (60km/h)', 11: 'No Left Turn', 17: 'No Blowing of Horns', 
           30: 'Bike Lane Ahead', 44: 'Side Road Junction Ahead (Left)'}

st.title('Traffic Sign Classification by Group 6')

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    image = Image.open(uploaded_file)
    image = preprocess_image(image)

    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index in classes:
        predicted_class_label = classes[predicted_class_index]
        st.success(f"Prediction: {predicted_class_label}")
    else:
        st.error("Unknown traffic sign class")
