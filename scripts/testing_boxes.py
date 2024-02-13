import numpy as np
import matplotlib.pyplot as plt
from utils import calc_iou
from fishnet_detector import FishnetDetector
import json
import os

def load_test_datapoint_and_prediction(test_img, detector, img_path="./data/foid_images_v100/images/"):
    '''
    Load a test image and predict on it
    Args:
        test_img: str, the name of the test image
        model_path: str, the path to the model
        img_path: str, the path to the test images
    Returns:
        ground_truths: list of tuples, the ground truth boxes
        predictions: list of tuples, the predicted boxes
    '''
    # get the test point
    one_test = test_data_dict[test_img]

    # Get the boxes from the first test image
    one_test_boxes = [item[:2] for item in one_test]  # List comprehension to get the boxes

    # x1, y1, x2, y2
    ground_truths = [(int(box[0][0]), int(box[1][0]), int(box[0][1]), int(box[1][1])) for box in one_test_boxes]
    # print('ground truths', ground_truths)
    
    # Get the predictions for the first test image
    image_path = os.path.join(img_path, f"{test_img}.jpg")
    pred_output = detector.detect(img_path=image_path, thresh_human=0, thresh_fish=0)
    pred_tensor = pred_output[0]['boxes']
    pred_conf = pred_output[0]['scores']
    
    # Convert the predictions to the format used in the ground truths
    predictions = [((int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())), float(score)) for box, score in zip(pred_tensor, pred_conf)]  # Placeholder for predictions and confidence scores
    # print('predictions', predictions)

    return ground_truths, predictions

def calculate_average_precision(predictions, ground_truths, iou_thresh=0.5, plot=True):
    """
    Calculate the Average Precision (AP) for a single image given predictions and ground truths.
    
    Parameters:
    - predictions: A list of tuples (box, confidence) for predicted bounding boxes and their confidence scores.
    - ground_truths: A list of bounding boxes for the ground truths.
    
    Returns:
    - AP: The Average Precision for the given predictions and ground truths.
    """
    # Sort predictions by descending confidence
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    TP, FP = 0, 0
    total_gt = len(ground_truths)
    matched_gt = set()
    precisions, recalls = [], []
    
    # Iterate through predictions and calculate precision and recall
    for pred_box, _ in sorted_preds:
        best_iou = 0
        best_gt_idx = None
        # Iterate through ground truths to find the best match for the prediction
        for gt_idx, gt_box in enumerate(ground_truths):
            iou = calc_iou(pred_box, gt_box)
            if iou > best_iou and gt_idx not in matched_gt:
                best_iou = iou
                best_gt_idx = gt_idx

        # Update TP and FP
        if best_iou >= iou_thresh:  # Using 0.5 as IoU threshold for a match
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1
        
        precision = TP / (TP + FP)
        recall = TP / total_gt
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP as the area under the PR curve
    # Sorting recalls since they might not be in order due to varying confidences
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    AP = np.trapz(sorted_precisions, sorted_recalls)

    if plot:
        # Plot the PR curve
        plt.plot(sorted_recalls, sorted_precisions)
        plt.fill_between(sorted_recalls, sorted_precisions, alpha=0.2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Average Precision: {AP:.4f}")
        plt.show()
    
    return AP

if __name__ == "__main__":
    # Load the testing JSON file
    test_data_dict = json.load(open("data/test_labels.json"))
    test_file_list = list(test_data_dict.keys())

    # Model
    # load Model
    detector = FishnetDetector(model_path="./app/models/best_model.pth")
    APs = []
    for i, test_img in enumerate(test_file_list):
        # print(f"Testing on {test_img}")
        # get the ground truths and predictions
        ground_truths, predictions = load_test_datapoint_and_prediction(test_img, detector)

        # Calculate AP for constant IoU threshold
        AP = calculate_average_precision(predictions, ground_truths, plot=False)
        APs.append(AP)
        # print(f"Average Precision: {AP:.4f}")
        if i % 20 == 0:
            print(f"Processed {i+1}/{len(test_file_list)} images...")

    # Calculate the mean AP
    mean_AP = np.mean(APs)
    print(f"Mean Average Precision: {mean_AP:.4f}")