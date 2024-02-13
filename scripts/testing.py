import json
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from fishnet_detector import FishnetDetector

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def format_model_output(model_output, score_threshold=0.8):
    formatted_output = []
    
    boxes = model_output[0]['boxes'].detach().cpu().numpy()
    scores = model_output[0]['scores'].detach().cpu().numpy()
    labels = model_output[0]['labels'].detach().cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            # Extract bounding box coordinates and round them
            x_min, y_min, x_max, y_max = map(int, box)
            class_label = COCO_INSTANCE_CATEGORY_NAMES[label]
            if class_label == 'person':
                class_label = 'Human'

            # Format according to your required JSON structure
            detection = [[[str(x_min), str(x_max)], [str(y_min), str(y_max)], class_label]]
            formatted_output.append(detection)

    return formatted_output


def format_best_model(model_output, score_threshold=0.8):
    formatted_output = []

    # Iterate through the detections
    for box, label, score in zip(model_output[0]['boxes'], model_output[0]['labels'], model_output[0]['scores']):
        score = score.item()  # Convert tensor to a Python float
        if score >= score_threshold:
            # Extract bounding box coordinates and round them
            x_min, y_min, x_max, y_max = map(int, box)
            
            # Format according to your required JSON structure
            detection = [[[str(x_min), str(x_max)], [str(y_min), str(y_max)], label]]
            formatted_output.append(detection)

    return formatted_output


def run_mean_model_inference_on_images(model, image_ids, image_folder):
    results = {}
    tensor_transform = transforms.ToTensor()
    count = 0
    for image_id in image_ids:
        count += 1
        if (count+1) % 20 == 0:
            # break
            print(f"Processed {count+1} images")
        # Assuming image files are named with their image IDs
        image_path = f"{image_folder}/{image_id}.jpg"  # Adjust this path format as needed
        image = Image.open(image_path)
        normalized_image = tensor_transform(image).unsqueeze(0)

        model_output = model(normalized_image)
        
        formatted_output = format_model_output(model_output)
        results[image_id] = [formatted_output]

    return results

def run_best_model_inference_on_images(model, image_ids, image_folder):
    results = {}
    count = 0
    for image_id in image_ids:
        count+=1
        if (count+1) % 20 == 0:
            print(f"Processed {count+1} images")

        # Assuming image files are named with their image IDs
        image_path = f"{image_folder}/{image_id}.jpg"  # Adjust this path format as needed
        image = Image.open(image_path)

        model_output = model.detect(image_path, thresh_human=0.8, thresh_fish=0.6)
        # print(model_output)
        formatted_output = format_best_model(model_output)
        results[image_id] = [formatted_output]

    return results


def mean_model_testing(img_folder):
    # Load the model
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    model.eval()

    # Load the test labels JSON
    with open('../data/test_labels.json', 'r') as file:
        test_labels = json.load(file)

    # Extract image IDs from the JSON
    image_ids = list(test_labels.keys())
    image_folder = img_folder  # Specify the folder where your images are stored

    # Run inference on the images
    inference_results = run_mean_model_inference_on_images(model, image_ids, image_folder)

    # Output the results as JSON
    with open('../data/mean_model_results.json', 'w') as file:
        json.dump(inference_results, file, indent=4)

def best_model_testing(img_folder):
    
        # Load the pre-trained model
    model = FishnetDetector(model_path="../data/best_model.pth")
    # Load the test labels JSON
    with open('../data/test_labels.json', 'r') as file:
        test_labels = json.load(file)

    # Extract image IDs from the JSON
    image_ids = list(test_labels.keys())
    image_folder = img_folder  # Specify the folder where your images are stored

    # Run inference on the images
    inference_results = run_best_model_inference_on_images(model, image_ids, image_folder)

    # Output the results as JSON
    with open('../data/best_model_results.json', 'w') as file:
        json.dump(inference_results, file, indent=4)



if __name__ == "__main__":
    #best_model_testing('../data/test_imgs')  
    mean_model_testing('../data/test_imgs')