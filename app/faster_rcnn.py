import os
import dotenv
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from . import IMG_FOLDER

# Original class to ID mapping
class_to_id = {'Human': 0, 'Swordfish': 1, 'Albacore': 2, 'Yellowfin tuna': 3, 'No fish': 4, 'Mahi mahi': 5, 'Skipjack tuna': 6, 'Unknown': 7, 'Wahoo': 8, 'Bigeye tuna': 9, 'Striped marlin': 10, 'Opah': 11, 'Blue marlin': 12, 'Escolar': 13, 'Shark': 14, 'Tuna': 15, 'Water': 16, 'Oilfish': 17, 'Pelagic stingray': 18, 'Marlin': 19, 'Great barracuda': 20, 'Shortbill spearfish': 21, 'Indo Pacific sailfish': 22, 'Lancetfish': 23, 'Long snouted lancetfish': 24, 'Black marlin': 25}
# Create inverse mapping from ID to class name
id_to_class = {v: k for k, v in class_to_id.items()}

# Load environment variables
dotenv.load_dotenv()

def load_fastercnn(model_weights_path):
    '''
    Load the Faster R-CNN model with a ResNet-50-FPN backbone.
    Args:
        model_weights_path: str, path to the model weights
    Returns:
        model: torch model, Faster R-CNN model
    '''
    # Load the model
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    # Change the final head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 26
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Set the model to evaluation mode
    model.eval()

    # Load the model weights if a path is provided
    if model_weights_path:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    
    return model

def plot_bounding_boxes(image_in, results, x_scale, y_scale, threshold=0.8):
    '''
    Plot the bounding boxes and labels of the detected objects.
    Args:
        image_in: np.array, input image
        results: dict, detection results
        x_scale: float, scaling factor for the x-axis
        y_scale: float, scaling factor for the y-axis
        threshold: float, threshold for the detection confidence
    Returns:
        np.array: image with bounding boxes and labels
    '''
    # Ensure results is not empty and contains the expected structure
    if not results or 'boxes' not in results[0] or 'labels' not in results[0] or 'scores' not in results[0]:
        print("Invalid results format.")
        return image_in
    
    # Get detections for the first image in the batch
    detections = results[0]
    
    ret_img = cv2.cvtColor(image_in, cv2.COLOR_RGB2BGR)
    # Plot bounding boxes and labels of the detected objects
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if score > threshold:
            # Convert tensor to numpy array and to integer coordinates
            box = box.cpu().numpy().astype(np.int32)
            label = label.item()  # Convert to Python scalar
            score = score.item()  # Convert to Python scalar
            
            # Scale the bounding boxes
            box = [int(box[0] * x_scale), int(box[1] * y_scale),
                int(box[2] * x_scale), int(box[3] * y_scale)]
            
            # Draw rectangle
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            color = (255, 0, 0)  # Red
            thickness = 2
            
            image_boxed = cv2.rectangle(image_in, start_point, end_point, color, thickness)
            
            # Get the class name
            class_name = id_to_class[label]
            # Label with class name and probability
            label_text = f"{class_name}: {score:.2f}"
            position = (box[0], box[1] - 10)  # Position for text is slightly above the top-left corner of the box
            font_scale = 0.5
            font_color = (255, 0, 0)  # Red 
            line_type = 1
            
            image_labeled = cv2.putText(image_boxed, label_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, line_type)
            ret_img = cv2.cvtColor(image_labeled, cv2.COLOR_RGB2BGR)
    return ret_img


def process_image(file, model_pth, thresh=0.5):
    '''
    Process the input image using the Faster R-CNN model.
    Args:
        file: str, path to the input image file
        model_pth: str, path to the model weights
        thresh: float, threshold for the detection confidence
    Returns:
        str: path to the processed image
    '''
    # load model
    model = load_fastercnn(model_pth)
    print(f"Model loaded from {model_pth}")
    # Load the image
    # Load the image with OpenCV
    image_cv = cv2.imread(file)
    original_dims = image_cv.shape[:2]
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
    
    # Calculate scale factors
    resized_dims = (800, 400)  # Width, Height
    x_scale = original_dims[1] / resized_dims[0]
    y_scale = original_dims[0] / resized_dims[1]

    # Plot the bounding boxes
    image = plot_bounding_boxes(image_cv_rgb, results, x_scale, y_scale, threshold=thresh)
    # Save the image
    input_file_name = os.path.basename(file).split('.')[0]
    image_name = f'{input_file_name}_processed.jpg'
    image_path = os.path.join(IMG_FOLDER, image_name)
    print(f"Saving processed image to {image_path}")
    cv2.imwrite(image_path, image)
    return image_name

