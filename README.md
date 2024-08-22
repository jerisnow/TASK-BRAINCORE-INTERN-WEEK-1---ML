# TASK-BRAINCORE-INTERN-WEEK-1---ML
Self Learning about Teachable Machine and YOLO

## Teachable Machine 
This repository contains a Keras model converted from Teachable Machine, along with the necessary labels file and a Python GUI script for using the model.

### Project Structure

```bash
.
├── Teachable Machine
│   ├── converted_keras
│   │   ├── keras_model.h5            # Exported model from Teachable Machine
│   │   └── labels.txt                # Class labels for the model
│   └── Gui_box.py                    # GUI application to test the model
├── YOLO_Object_Detection
│   ├── hasil train                   # Training results folder
│   ├── model evaluation results      # Evaluation results from testing
│   ├── model evaluation tests        # Testing and evaluation scripts
│   ├── best_model.pt                 # YOLO model's best weights
│   ├── evaluate_model.py             # Model evaluation script
│   ├── gui_predict_yolo.py           # GUI to test YOLO object detection
│   ├── metrics_model.py              # Script for calculating metrics
│   ├── results.png                   # Example result image
│   ├── train_yolo.py                 # Script to train YOLO model
│   └── yolov8n.pt                    # YOLOv8 weights
