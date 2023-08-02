import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model_path = 'full_model.h5'
model = tf.keras.models.load_model(model_path)

# Define a function to make predictions on uploaded images


def predict(image):
    # Preprocess the image to match the model's input shape
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0  # Normalize
    image_array = image_array[np.newaxis, ...]  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    return predicted_class


st.title("Image Classifier")
st.write("Upload an image and the model will predict the class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the class of the uploaded image
    predicted_class = predict(image)

    # You may map this to a human-readable label
    st.write("Prediction:", predicted_class)
