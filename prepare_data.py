import os
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. Configuration ---
IMAGE_FOLDER_PATH = "images" 
IMAGE_SIZE = (256, 256)

# --- 2. Load Images and Create Labels from Folders ---
print("--- Step 1: Finding breed folders and processing images ---")
training_data = []
supported_formats = ('.jpg', '.jpeg', '.png') # We now accept multiple formats

try:
    breed_folders = os.listdir(IMAGE_FOLDER_PATH)
    print(f"Found {len(breed_folders)} breed folders.")
except FileNotFoundError:
    print(f"ERROR: The folder '{IMAGE_FOLDER_PATH}' was not found.")
    exit()

for breed in breed_folders:
    folder_path = os.path.join(IMAGE_FOLDER_PATH, breed)
    class_label = breed
    
    if not os.path.isdir(folder_path):
        continue

    for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {breed}"):
        # Check if the file has a supported extension
        if img_name.lower().endswith(supported_formats):
            try:
                img_path = os.path.join(folder_path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, IMAGE_SIZE)
                training_data.append([resized_array, class_label])
            except Exception as e:
                pass # Skip corrupted images

# --- 3. Save to NumPy Arrays ---
print("\n--- Step 2: Shuffling and saving the data... ---")
np.random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

np.save("processed_data/features.npy", X)
np.save("processed_data/labels.npy", y)

print("\nData preparation complete!")
print(f"Total images processed: {len(X)}")
print("Ready for re-training!")