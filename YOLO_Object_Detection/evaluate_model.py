from ultralytics import YOLO
import matplotlib.pyplot as plt


model = YOLO('/content/best_model.pt')

image_paths = [
    '/content/test1.jpg',
    '/content/test2.jpg',
    '/content/test3.jpg',
    '/content/test4.jpg',
    '/content/test5.jpg'
]

for i, image_path in enumerate(image_paths):
    results = model(image_path)
    
    results[0].show()
    
    img_with_boxes = results[0].plot() 
    plt.figure(figsize=(10, 6)) 
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.title(f'Object Detection Result for Images{i+1}')
    plt.show()

