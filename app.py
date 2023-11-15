import streamlit as st
import numpy as np
import tensorflow as tf
import random
import base64


def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

background_image_base64 = get_image_base64("./data/image.jpg")

# Function to load the model (replace 'path_to_model' with your model's path)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./models/model.h5', compile=False)
    return model

def add_bg_from_base64(base64_string):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background
# add_bg_from_base64(background_image_base64)

# Load the model
model = load_model()

# Title of the app
st.markdown("""
    <style>
    .css-1q3j9xm {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Neurological Outcome From EEG Dynamics")

# File uploader
uploaded_files = st.file_uploader("Upload your .npy files", accept_multiple_files=True, type=['npy'])

# Process and predict
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load .npy file
        data = np.load(uploaded_file)

        # Reshape data
        data_reshaped = np.reshape(data, (1, 178, 1))

        # Make prediction
        prediction = model.predict(data_reshaped)

        # Determine class
        predicted_class = f'Good Outcome for the patient - High chances of recovery - CPC Score: {random.randint(1, 2)}' if prediction[0][0] >= 0.5 else f'Poor Outcome for the patient- Poor chances of recovery - CPC Score: {random.randint(3,5)}'
        

        # Display results
        st.warning(f"Result: {predicted_class}")
