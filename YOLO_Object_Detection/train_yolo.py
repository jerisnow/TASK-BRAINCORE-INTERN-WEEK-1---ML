from ultralytics import YOLO
import os

dataset_path = '/content/Project_Furniture-8'

# Verify dataset contents
print("Contents of dataset directory:")
print(os.listdir(dataset_path))

data_yaml_path = '/content/Project_Furniture-8/data.yaml'

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model with fewer epochs and batch size adjustments
model.train(
    data=data_yaml_path,
    epochs=50,            # Reduced number of epochs
    batch=16,             # Batch size
    imgsz=640,            # Image size
    name='train',
    patience=10,          # Early stopping if no improvement after 10 epochs
    lr0=0.01,             # Initial learning rate
    save_period=10       # Save checkpoint every 10 epochs
)

model.save('best_model.pt')
