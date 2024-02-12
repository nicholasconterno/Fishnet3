import json
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def format_model_output(model_output, score_threshold=0.8):
    formatted_output = []
    
    boxes = model_output[0]['boxes'].detach().cpu().numpy()
    scores = model_output[0]['scores'].detach().cpu().numpy()

    for box, score in zip(boxes, scores):
        if score >= score_threshold:
            # Extract bounding box coordinates and round them
            x_min, y_min, x_max, y_max = map(int, box)

            # Format according to your required JSON structure
            # Assuming the class is always "Human" as per your previous JSON
            # Modify this if you have class labels
            detection = [[[str(x_min), str(x_max)], [str(y_min), str(y_max)], "Human"]]
            formatted_output.append(detection)

    return formatted_output

def run_inference_on_images(model, image_ids, image_folder):
    results = {}
    tensor_transform = transforms.ToTensor()
    count = 0
    for image_id in image_ids:
        count += 1
        if count % 50 == 0:
            print(f"Processed {count} images")
        # Assuming image files are named with their image IDs
        image_path = f"{image_folder}/{image_id}.jpg"  # Adjust this path format as needed
        image = Image.open(image_path)
        normalized_image = tensor_transform(image).unsqueeze(0)

        model_output = model(normalized_image)
        formatted_output = format_model_output(model_output)
        results[image_id] = [formatted_output]

    return results

def mean_model_testing(img_folder):
    # Load the model
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    model.eval()

    # Load the test labels JSON
    with open('test_labels.json', 'r') as file:
        test_labels = json.load(file)

    # Extract image IDs from the JSON
    image_ids = list(test_labels.keys())
    image_folder = img_folder  # Specify the folder where your images are stored

    # Run inference on the images
    inference_results = run_inference_on_images(model, image_ids, image_folder)

    # Output the results as JSON
    with open('mean_model_results.json', 'w') as file:
        json.dump(inference_results, file, indent=4)

def best_model_testing(img_folder):
    
        # Load the pre-trained model
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")

    # # Freeze all model parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # Replace the box predictor (classifier head)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 26  # Update this if you have a different number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load your saved model weights for CPU
    path_to_saved_weights = '../data/best_model.pth'  # Update this path
    model.load_state_dict(torch.load(path_to_saved_weights, map_location=torch.device('cpu')))

    # Your model is now ready to be used with the updated weights and box predictor
    model.eval()
    # Load the test labels JSON
    with open('test_labels.json', 'r') as file:
        test_labels = json.load(file)

    # Extract image IDs from the JSON
    image_ids = list(test_labels.keys())
    image_folder = img_folder  # Specify the folder where your images are stored

    # Run inference on the images
    inference_results = run_inference_on_images(model, image_ids, image_folder)

    # Output the results as JSON
    with open('best_model_results.json', 'w') as file:
        json.dump(inference_results, file, indent=4)


mean_model_testing()