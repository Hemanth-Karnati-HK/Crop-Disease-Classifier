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
    image_array = np.array(image_resized) / 255.0
    image_array = image_array[np.newaxis, ...]

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    # Return both class index and disease name
    return predicted_class, index_to_class[str(predicted_class)]

def translate_text(description):
    st.write("Translate Description:")
    # User input for the text to translate
    target_language = st.selectbox("Select the target language:", ('', 'Telugu', 'Tamil', 'Hindi'))

    # Language codes corresponding to the selected languages
    language_codes = {'Telugu': 'te', 'Tamil': 'ta', 'Hindi': 'hi'}

    # Create a URL to translate the text using Google Translate
    if target_language:
        language_code = language_codes[target_language]
        description_truncated = description[:100]
        translate_url = f"https://translate.google.com/?sl=auto&tl={language_code}&text={description_truncated}&op=translate"
        st.markdown(f"[Click here to translate the text into {target_language} using Google Translate]({translate_url})")

def display_prediction(predicted_disease_name, disease_descriptions):
    description = disease_descriptions.get(predicted_disease_name, "Description not available")
    st.write("Prediction:", predicted_disease_name)
    st.subheader("Description")
    st.write(description.split("Prevention")[0])

    if "Prevention" in description:
        st.subheader("Prevention")
        prevention_description = description.split("Prevention")[1]
        st.write(prevention_description)

    st.write("If you have any concerns or need further assistance, please consult with a local agricultural expert.")
    
    # Add a button to trigger the translation section
    if st.button('Translate Description'):
        translate_text(description)

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

    # Display the prediction
    display_prediction(predicted_disease_name, disease_descriptions)
