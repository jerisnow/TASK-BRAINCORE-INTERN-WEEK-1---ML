import os
import cv2
import torch
from ultralytics import YOLO
import PySimpleGUI as sg

# Ensure the model file exists
model_path = '/content/best_model.pt'

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the trained YOLOv8 model
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

# Define object labels to display
object_labels = ['Car', 'Chair', 'Door', 'Person', 'Table']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to process video frames
def process_frame(window):
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        boxes = results[0].boxes  # Get detection boxes

        # Display the frame with predictions
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = object_labels[cls]

            if label in object_labels:
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame in a window
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        if window['image']:
            window['image'].update(data=imgbytes)
        
    

# Create the layout for the GUI
layout = [
    [sg.Text('Object Detection', size=(30, 1), justification='center', font='Helvetica 20')],
    [sg.Image(filename='', key='image')],
    [sg.Button('Start', size=(10, 1), font='Helvetica 14'), sg.Button('Stop', size=(10, 1), font='Helvetica 14')]
]

# Create the main window
window = sg.Window('Object Detection with YOLO', layout, location=(800, 400))

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
