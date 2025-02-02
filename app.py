import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("model.keras")

# Emotion labels
labels = ['sad', 'disgust', 'fear', 'happy', 'neutral', 'angry', 'surprise']

# App title and description with styling
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #FF4B4B;">🎭 Emotion Classification App 🎭</h1>
        <p style="color: #666666;">Upload an image to detect the emotion displayed on the face.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image to classify emotion (JPG, PNG, JPEG):", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale

    # Resize the image for display while maintaining aspect ratio
    max_width = 300  # Define the maximum width for display
    original_width, original_height = image.size
    aspect_ratio = original_height / original_width
    display_height = int(max_width * aspect_ratio)
    resized_image = image.resize((max_width, display_height))

    # Center the image using columns
    col1, col2, col3 = st.columns([1, 2, 1])  # Add 3 columns with col2 being the middle one
    with col2:
        st.image(resized_image, caption="Uploaded Image (Resized for Display)", use_container_width=True)

    # Add a spinner while processing
    with st.spinner("Processing the image..."):
        # Preprocess the image for prediction
        preprocessed_image = image.resize((48, 48))  # Resize to 48x48 pixels
        image_array = img_to_array(preprocessed_image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(image_array)
        emotion = labels[np.argmax(predictions)]

    # Display the result
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="color: #4CAF50;">Prediction Complete!</h2>
            <h3>The predicted emotion is: <span style="color: #FF5722;">{emotion}</span></h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <p style="color: #999999;">👆 Upload an image to get started!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
