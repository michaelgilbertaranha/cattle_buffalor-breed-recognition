import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# --- 1. Configuration ---
# The path to your saved model, downloaded from Kaggle
MODEL_PATH = 'breed_classifier1.h5'

# The path to the new image you want to test
IMAGE_TO_TEST = 'test_image.jpeg' # <-- IMPORTANT: CHANGE this to your test image's name

# The labels file is needed to know the breed names
LABELS_FILE = 'processed_data/labels.npy'

# The size must match the size used during training
IMAGE_SIZE = (256, 256) 

# --- 2. Load the Model and Labels ---
print("--- Step 1: Loading the trained model and labels ---")
try:
    model = load_model(MODEL_PATH)
    # allow_pickle=True is needed for loading string arrays with np.load
    labels = np.load(LABELS_FILE, allow_pickle=True)
    # Get the unique breed names in alphabetical order, which matches the model's output
    class_names = sorted(list(set(labels)))
except Exception as e:
    print(f"ERROR: Could not load model or labels. Make sure '{MODEL_PATH}' and '{LABELS_FILE}' exist.")
    print(f"Error details: {e}")
    exit()

print(f"Model '{MODEL_PATH}' loaded successfully.")
print(f"Classifier is ready to identify {len(class_names)} breeds.")

# --- 3. Load and Prepare the Test Image ---
print(f"\n--- Step 2: Loading and preparing '{IMAGE_TO_TEST}' ---")
try:
    image = cv2.imread(IMAGE_TO_TEST)
    # Resize it to the same size as our training images
    image_resized = cv2.resize(image, IMAGE_SIZE)
    # Normalize pixel values from 0-255 to 0-1
    image_normalized = image_resized.astype("float32") / 255.0
    # Add an extra dimension because the model expects a "batch" of images
    image_batch = np.expand_dims(image_normalized, axis=0)
except Exception as e:
    print(f"ERROR: Could not load or process the test image. Check the file name and path.")
    print(f"Error details: {e}")
    exit()

# --- 4. Make a Prediction ---
print("--- Step 3: Making a prediction ---")
predictions = model.predict(image_batch)

# Find the highest confidence score and the corresponding breed name
confidence = np.max(predictions)
predicted_class_index = np.argmax(predictions)
predicted_breed = class_names[predicted_class_index]

print(f"\n--> Prediction: {predicted_breed}")
print(f"--> Confidence: {confidence:.2%}")

# --- 5. Display the Result (with Confidence Threshold) ---
CONFIDENCE_THRESHOLD = 0.70 # 70%

if confidence > CONFIDENCE_THRESHOLD:
    result_text = f"{predicted_breed} ({confidence:.2%})"
    text_color = (0, 255, 0) # Green for high confidence
else:
    result_text = "Breed Not Recognized"
    text_color = (0, 0, 255) # Red for low confidence

# Put the text on the original image to display
# Increase font size and thickness for better visibility
cv2.putText(image, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

# Show the final image
cv2.imshow("Prediction Result", image)
print("\nDisplaying result. Press any key on the image window to close it.")
cv2.waitKey(0)
cv2.destroyAllWindows()