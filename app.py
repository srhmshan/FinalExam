import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

model = load_model('tsr_model.h5')

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classes = {5: 'Speed limit (60km/h)', 14: 'No Left Turn', 27: 'No Blowing of Horns', 
           29: 'Bike Lane Ahead', 39: 'Side Road Junction Ahead (Left)'}

st.title('Traffic Sign Recognition by Group 6')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    image = Image.open(uploaded_image)
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred_probabilities = model.predict(image)
    pred_class_index = np.argmax(pred_probabilities, axis=1)[0]
    sign = classes[pred_class_index]
    st.write(f"Predicted Sign: {sign}")
