# TASK-BRAINCORE-INTERN-WEEK-1---ML
Self Learning about Teachable Machine and YOLO

### Project Structure

```bash
.
├── Teachable Machine
│   ├── converted_keras
│   │   ├── keras_model.h5            # Exported model keras from Teachable Machine
│   │   └── labels.txt                # Class labels for the model
│   └── Gui_box.py                    # Implementation using GUI application to test the model
├── YOLO_Object_Detection
│   ├── hasil train                   # Training results folder
│   ├── model evaluation results      # Evaluation results from testing
│   ├── model evaluation tests        # Testing and evaluation images
│   ├── best_model.pt                 # YOLO model's best weights
│   ├── evaluate_model.py             # Model evaluation script
│   ├── gui_predict_yolo.py           # Implementation model YOLO using GUI application to test YOLO object detection
│   ├── metrics_model.py              # Script for calculating metrics
│   ├── results.png                   # Result image matrics
│   ├── train_yolo.py                 # Script to train YOLO model
│   └── yolov8n.pt                    # YOLOv8 weights
.
```

## Teachable Machine 
This repository contains a Keras model converted from Teachable Machine, along with the necessary labels file and a Python GUI script for implement the model.

### Implementation Teachable Machine Model using Gui 
To test the the Keras model real-time using a graphical interface:
```
python Gui_box.py
```
Link Screen Record of The Prediction Results:
https://drive.google.com/file/d/1FnlPGHRd8v5QhE-8JpWsHb9B1xV95aP8/view?usp=sharing 


## YOLO
This repository contains the implementation of a YOLO (You Only Look Once) object detection system and training object detection usinh YOLO (You Only Look Once).

### Implementation YOLO Model using GUI

To test the trained YOLO model real-time using a graphical interface:
```
python gui_predict_yolo.py
```
Link Screen Record of The Prediction Results:
https://drive.google.com/file/d/106E5TlCJ5GGoElCD8wHGB0uB8FintUJu/view?usp=sharing 
