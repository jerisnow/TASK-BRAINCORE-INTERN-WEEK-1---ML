import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import PySimpleGUI as sg
from PIL import Image, ImageTk

# Ensure the model file exists
model_path = 'keras_model.h5'
labels_path = 'labels.txt'

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.isfile(labels_path):
    raise FileNotFoundError(f"Labels file not found: {labels_path}")

# Load the model
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

# Load the labels
with open(labels_path, 'r') as f:
    object_labels = [line.strip() for line in f.readlines()]

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to preprocess and predict
def preprocess_and_predict(frame):
    image = cv2.resize(frame, (224, 224))
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1  # Normalize

    # Predict using the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    label = object_labels[index]
    confidence_score = prediction[0][index]

    return label, confidence_score

# Function to process video frames
def process_frame(window):
    ret, frame = cap.read()
    if ret:
        label, confidence_score = preprocess_and_predict(frame)

        # Draw bounding box and label on the frame
        height, width, _ = frame.shape
        x1, y1, x2, y2 = int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9)  # Example bounding box

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label and confidence score
        label_text = f"{label} {confidence_score:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert frame to ImageTk format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the window image element
        if window['image']:
            window['image'].update(data=imgtk)

# Create the layout for the GUI
layout = [
    [sg.Text('Object Detection with Teachable Machine', size=(30, 1), justification='center', font='Helvetica 20')],
    [sg.Image(filename='', key='image')],
    [sg.Button('Start', size=(10, 1), font='Helvetica 14'), sg.Button('Stop', size=(10, 1), font='Helvetica 14')]
]

# Create the main window
window = sg.Window('Object Detection with Teachable Machine', layout, location=(800, 400))

# Event loop to process "Start" and "Stop" button clicks
start_processing = False
while True:
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED or event == 'Stop':
        break
    elif event == 'Start':
        start_processing = True
    
    if start_processing:
        process_frame(window)

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
window.close()

