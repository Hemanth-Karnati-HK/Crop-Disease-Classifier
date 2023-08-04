import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import urllib.parse

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
    image_array = np.array(image_resized) / 255.0
    image_array = image_array[np.newaxis, ...]
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, index_to_class[str(predicted_class)]

def translate_text(description):
    target_language = st.selectbox("Select the target language:", ('', 'Telugu', 'Tamil', 'Hindi'))
    language_codes = {'Telugu': 'te', 'Tamil': 'ta', 'Hindi': 'hi'}

    if target_language:
        language_code = language_codes[target_language]
        encoded_text = urllib.parse.quote(description)
        translate_url = f"https://translate.google.com/?sl=auto&tl={language_code}&text={encoded_text}&op=translate"
        st.markdown(f"[Click here to translate the text into {target_language} using Google Translate]({translate_url})")

def display_prediction(predicted_disease_name, disease_descriptions):
    description = disease_descriptions.get(predicted_disease_name, "Description not available")
    st.write("Prediction:", predicted_disease_name)
    st.subheader("Description")
    if "Prevention" in description:
        description_parts = description.split("Prevention")
        st.write(description_parts[0])
        st.subheader("Prevention")
        st.write(description_parts[1])
    else:
        st.write(description)

    st.write("If you have any concerns or need further assistance, please consult with a local agricultural expert.")
    translate_text(description)


st.title("Farm Disease Identifier")
st.write("Welcome! Upload an image of a plant, and I'll identify the disease (if any). Together, we can ensure a healthy crop!")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class, predicted_disease_name = predict(image)
    display_prediction(predicted_disease_name, disease_descriptions)
