import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import urllib.parse
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import tempfile

# Set page title
st.set_page_config(page_title="Farm Disease Identifier")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f0f5;
    }
    h1 {
        color: #2e7d32;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0
    image_array = image_array[np.newaxis, ...]

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, index_to_class[str(predicted_class)]

def transcribe_audio(audio_bytes):
    # Convert bytes to audio
    audio_file = BytesIO(audio_bytes)
    audio = AudioSegment.from_wav(audio_file)

    # Use a temporary file for use with SpeechRecognition
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_filename = temp_file.name
        audio.export(temp_filename, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_filename) as source:
        recorded_audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(recorded_audio)  # You can use other engines, like Sphinx, here
        except sr.UnknownValueError:
            text = "Sorry, I could not understand the audio."
        except sr.RequestError:
            text = "Could not request results; check your network connection."

    # Remove temporary file
    os.remove(temp_filename)

    return text

def display_prediction(predicted_disease_name, disease_descriptions):
    description = disease_descriptions.get(predicted_disease_name, "Description not available")
    st.write("Prediction:", predicted_disease_name)
    st.subheader("Description")
    st.write(description.split("Prevention")[0])

    if "Prevention" in description:
        prevention_description = description.split("Prevention")[1]
        st.subheader("Prevention")
        st.write(prevention_description)
    
    st.write("If you have any concerns or need further assistance, please consult with a local agricultural expert.")

    # Translate
    st.subheader("Translate the Description")
    target_language = st.selectbox("Select the target language:", ('', 'Telugu', 'Tamil', 'Hindi'))
    language_codes = {'Telugu': 'te', 'Tamil': 'ta', 'Hindi': 'hi'}
    
    if target_language:
        language_code = language_codes[target_language]
        description_for_translation = description
        encoded_description = urllib.parse.quote_plus(description_for_translation)
        translate_url = f"https://translate.google.com/?sl=auto&tl={language_code}&text={encoded_description}&op=translate"
        st.markdown(f"[Click here to translate the text into {target_language} using Google Translate]({translate_url})")

st.title("🌱 Farm Disease Identifier 🌾")
st.write("Welcome to the Farm Disease Identifier! Upload an image of a plant, and I'll identify the disease (if any). Together, we can ensure a healthy crop! 🌻")

# Upload image
uploaded_file = st.file_uploader("Choose an image (jpg or png)...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying... 🧠")

    # Predict the class of the uploaded image
    predicted_class, predicted_disease_name = predict(image)

    # Display human-readable label and detailed description
    display_prediction(predicted_disease_name, disease_descriptions)

st.title("🌱 Farm Disease Identifier & Voice Assistant 🌾")

# Record audio
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.write("Transcribing... 🎤")
    transcription = transcribe_audio(audio_bytes)
    st.subheader("Transcription:")
    st.write(transcription)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for farmers everywhere!")
