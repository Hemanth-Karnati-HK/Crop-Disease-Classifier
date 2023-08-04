import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Load the trained model
model_path = 'full_model.h5'
model = tf.keras.models.load_model(model_path)

# Load class indices
with open('class_indices.json', 'r') as f:
    index_to_class = json.load(f)

# Load disease descriptions
with open('cure.json', 'r') as f:
    disease_descriptions = json.load(f)

def predict(image):
    # Preprocess the image to match the model's input shape
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0  # Normalize
    image_array = image_array[np.newaxis, ...]  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    # Return both class index and disease name
    return predicted_class, index_to_class[str(predicted_class)]

st.title("Farm Disease Identifier")
st.write("Welcome! Upload an image of a plant, and I'll identify the disease (if any). Together, we can ensure a healthy crop!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the class of the uploaded image
    predicted_class, predicted_disease_name = predict(image)

    # Display human-readable label and detailed description
    st.write("Prediction:", predicted_disease_name)
    st.write(disease_descriptions.get(predicted_disease_name, "Description not available"))
    st.write("If you have any concerns or need further assistance, please consult with a local agricultural expert.")
