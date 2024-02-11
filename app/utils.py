import os
import dotenv
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Load environment variables
dotenv.load_dotenv()

def load_model(model_weights_path=None):
    # Load the model
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    # Change the final head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 26
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Set the model to evaluation mode
    model.eval()

    # TODO: Load the model weights if a path is provided
    if model_weights_path:
        model.load_state_dict(torch.load(model_weights_path))
    
    return model

import cv2
import numpy as np

def plot_bounding_boxes(image_in, results):
    # Ensure results is not empty and contains the expected structure
    if not results or 'boxes' not in results[0] or 'labels' not in results[0] or 'scores' not in results[0]:
        print("Invalid results format.")
        return image_in
    
    # Get detections for the first image in the batch
    detections = results[0]
    
    # Plot bounding boxes and labels of the detected objects
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        # Convert tensor to numpy array and to integer coordinates
        box = box.cpu().numpy().astype(np.int32)
        label = label.item()  # Convert to Python scalar
        score = score.item()  # Convert to Python scalar
        
        # Draw rectangle
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        color = (255, 0, 0)  # Blue color in both BGR and RGB
        thickness = 2
        
        image_boxed = cv2.rectangle(image_in, start_point, end_point, color, thickness)
        
        # Label with class name and probability
        label_text = f"{label}: {score:.2f}"
        position = (box[0], box[1] - 10)  # Position for text is slightly above the top-left corner of the box
        font_scale = 0.5
        font_color = (255, 0, 0)  # Blue color in both BGR and RGB
        line_type = 2
        
        image_labeled = cv2.putText(image_boxed, label_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, line_type)

    return image_labeled


def process_image(file, temp_folder_path):
    # load model
    model = load_model()
    # Load the image
    # Load the image with OpenCV
    image_cv = cv2.imread(file)
    # Convert BGR (OpenCV default) image to RGB
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # Convert the image to a PIL Image
    image_pil = Image.fromarray(image_cv_rgb)
    
    # Preprocess the image
    # Define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((400, 800))
    ])
    # Apply the transforms
    image_tensor = transform(image_pil).unsqueeze(0)

    # Run the model
    with torch.no_grad():  # Ensure gradients are not computed
        results = model(image_tensor)
    
    # print(results)
    # Plot the bounding boxes
    image = plot_bounding_boxes(image_cv_rgb, results)
    # Save the image
    input_file_name = os.path.basename(file).split('.')[0]
    image_name = f'{input_file_name}_processed.jpg'
    image_path = os.path.join(temp_folder_path, image_name)
    print(f"Saving processed image to {image_path}")
    cv2.imwrite(image_path, image)
    return image_path, image_name

