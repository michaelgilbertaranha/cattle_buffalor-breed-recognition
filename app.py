import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# --- Configuration ---
MODEL_PATH = 'breed_classifier1.h5'
LABELS_FILE = 'labels.npy'
IMAGE_SIZE = (256, 256)

# --- Caching the model and labels for better performance ---
@st.cache_resource
def load_our_model():
    """Load the trained model and class names."""
    try:
        model = load_model(MODEL_PATH)
        labels = np.load(LABELS_FILE, allow_pickle=True)
        class_names = sorted(list(set(labels)))
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model or labels: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess the uploaded image to fit the model's input requirements."""
    # Convert PIL Image to OpenCV format (NumPy array)
    img_array = np.array(image)
    # Convert from RGB (PIL) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize and normalize
    resized_img = cv2.resize(img_bgr, IMAGE_SIZE)
    normalized_img = resized_img.astype("float32") / 255.0
    
    # Add a batch dimension
    return np.expand_dims(normalized_img, axis=0)

# --- Streamlit App UI ---
st.title("🐄 Buffalo & Cattle Breed Identifier")
st.write("Upload an image of a cow or buffalo to classify its breed.")

# Load the model and labels from the cache
model, class_names = load_our_model()

if model is not None:
    # Create the file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)

        # Get the top prediction
        confidence = np.max(predictions)
        predicted_class_index = np.argmax(predictions)
        predicted_breed = class_names[predicted_class_index]
        
        # Display the result
        st.write(f"### Prediction: *{predicted_breed}*")
        st.write(f"#### Confidence: *{confidence:.2%}*")
        
        # Add a confidence bar

        st.progress(float(confidence))
