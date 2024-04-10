import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Load the trained model
model = load_model('BrainTumor10Epochs-1.h5')

# Function to preprocess image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    img = cv2.resize(img, (64, 64))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)

# Function to predict brain tumor
def predict_tumor(image_path):
    input_img = preprocess_image(image_path)
    result = model.predict(input_img)
    return result

# Function to open file dialog and get image path
def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300), 'Antialias')  # Resizing with anti-aliasing
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to prevent garbage collection
        result = predict_tumor(file_path)
        result_label.config(text="Prediction: {}".format(result))

# Create the main application window
root = tk.Tk()
root.title("Brain Tumor Detection")

# Create a button to browse for image
browse_button = tk.Button(root, text="Browse Image", command=browse_file)
browse_button.pack(pady=10)

# Create label to display selected image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Create label to display prediction result
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the application
root.mainloop()
