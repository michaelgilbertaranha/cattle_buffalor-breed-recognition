import tensorflow as tf

# --- 1. Configuration ---
# Path to your saved Keras model
KERAS_MODEL_PATH = 'breed_classifier1.h5'

# Path where you want to save the converted TFLite model
TFLITE_MODEL_PATH = 'breed_classifier1.tflite'

print(f"--- Loading Keras model from: {KERAS_MODEL_PATH} ---")

# --- 2. Load the Keras Model ---
try:
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load the Keras model. Make sure the file exists.")
    print(f"Error details: {e}")
    exit()

# --- 3. Convert the Model to TensorFlow Lite ---
print(f"\n--- Converting model to TensorFlow Lite format... ---")
# Create a TFLite converter object from our Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations (optional but recommended for mobile)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Perform the conversion
tflite_model = converter.convert()
print("Model converted successfully.")

# --- 4. Save the TFLite Model ---
print(f"\n--- Saving TFLite model to: {TFLITE_MODEL_PATH} ---")
try:
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print("\nConversion complete!")
    print(f"Your model is now ready for Flutter at: {TFLITE_MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not save the TFLite file.")
    print(f"Error details: {e}")