import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

def load_resources():
    # Load the trained model
    model_path = 'full_model.h5'
    model = tf.keras.models.load_model(model_path)

    # Load class indices
    with open('class_indices.json', 'r') as f:
        index_to_class = json.load(f)

    # Load disease descriptions
    with open('cure.json', 'r') as f:
        disease_descriptions = json.load(f)

    return model, index_to_class, disease_descriptions

def preprocess_image(image):
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0
    return image_array[np.newaxis, ...]

def predict(image, model, index_to_class):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, index_to_class[str(predicted_class)]

def display_prediction(predicted_disease_name, disease_descriptions):
    description = disease_descriptions.get(predicted_disease_name, "Description not available")
    st.write("Prediction:", predicted_disease_name)
    st.subheader("Description")
    st.write(description.split("Prevention")[0])
    
    st.subheader("Prevention")
    st.write(description.split("Prevention")[1])
    st.write("If you have any concerns or need further assistance, please consult with a local agricultural expert.")

def main():
    st.title("Farm Disease Identifier")
    st.write("Welcome! Upload an image of a plant, and I'll identify the disease (if any). Together, we can ensure a healthy crop!")

    model, index_to_class, disease_descriptions = load_resources()

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")
        predicted_class, predicted_disease_name = predict(image, model, index_to_class)
        display_prediction(predicted_disease_name, disease_descriptions)

if __name__ == '__main__':
    main()
