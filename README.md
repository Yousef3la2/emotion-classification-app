# Emotion Classification App

## Overview
The **Emotion Classification App** is a web application built using Streamlit that can detect emotions from facial images. The app uses a pre-trained deep learning model to predict the emotion displayed on the face in the uploaded image. Supported emotions include:
- Sad
- Disgust
- Fear
- Happy
- Neutral
- Angry
- Surprise

The model is based on a deep convolutional neural network trained on 48x48 grayscale images, and the web app allows users to upload their images for emotion detection.

## Requirements
Before running the app, you need to install the following dependencies:

- **Streamlit**: For building the web application.
- **TensorFlow**: For using the pre-trained model.
- **NumPy**: For numerical operations.
- **Pillow**: For image processing.

You can install all dependencies by running:

```bash
pip install -r requirements.txt
