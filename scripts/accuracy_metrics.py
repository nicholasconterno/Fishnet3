import json
from collections import Counter
import numpy as np
'''
calculate_f1_score(true_labels, predicted_labels)
    Calculate the F1 score for a set of true and predicted labels.

    Args:
    - true_labels: list, the true labels.
    - predicted_labels: list, the predicted labels.

    Returns:
    - f1_score: float, the F1 score.

'''
def calculate_f1_score(true_labels, predicted_labels):
    true_counts = Counter(true_labels)
    predicted_counts = Counter(predicted_labels)
    labels = set(true_counts.keys()) | set(predicted_counts.keys())
    f1_scores = []

    for label in labels:
        TP = min(true_counts[label], predicted_counts[label])
        FP = predicted_counts[label] - TP
        FN = true_counts[label] - TP

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0

'''
load_labels_from_json(json_file)
    Load labels from a JSON file.

    Args:
    - json_file: str, the path to the JSON file.

    Returns:
    - data: dict, the labels data.
    '''
def load_labels_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

'''
calculate_batch_f1_score_from_json(true_labels_json, predicted_labels_json)
    
    Calculate the batch F1 score for a set of true and predicted labels.
    
    Args:
    - true_labels_json: str, the path to the JSON file containing the true labels.
    - predicted_labels_json: str, the path to the JSON file containing the predicted labels.
    
    Returns:
    - overall_f1_score: float, the overall F1 score.
    '''
def calculate_batch_f1_score_from_json(true_labels_json, predicted_labels_json):
    true_labels_data = load_labels_from_json(true_labels_json)
    predicted_labels_data = load_labels_from_json(predicted_labels_json)

    image_ids = set(true_labels_data.keys()) | set(predicted_labels_data.keys())
    total_f1_score = 0

    for image_id in image_ids:
        true_labels = true_labels_data.get(image_id, [])
        predicted_labels = predicted_labels_data.get(image_id, [])
        total_f1_score += calculate_f1_score(true_labels, predicted_labels)

    overall_f1_score = total_f1_score / len(image_ids) if image_ids else 0
    return overall_f1_score


'''
check_same_keys(json1, json2)
    Check if two JSON files have the same keys.

    Args:
    - json1: str, the path to the first JSON file.
    - json2: str, the path to the second JSON file.

    Returns:
    - bool, True if the JSON files have the same keys, False otherwise.'''
def check_same_keys(json1, json2):
    with open(json1, 'r') as file:
        data1 = json.load(file)
    with open(json2, 'r') as file:
        data2 = json.load(file)
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    return keys1 == keys2


'''
convert_to_box_coordinates_format(input_file, output_file)
    Convert a JSON file of detection data to a box coordinates format.

    Args:
    - input_file: str, the path to the input JSON file.
    - output_file: str, the path to the output JSON file.
    '''
def convert_to_box_coordinates_format(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    box_coordinates_format = {}
    for image_id, detections in data.items():
        boxes = []
        for detection_group in detections:
            for detection in detection_group:
                for det in detection:
                    if len(det) >= 2:  # To ensure it includes box coordinates
                        box_coordinates = det[0] + det[1]  # Concatenating x and y coordinates
                        boxes.append(box_coordinates)
        box_coordinates_format[image_id] = boxes

    with open(output_file, 'w') as file:
        json.dump(box_coordinates_format, file, indent=4)

'''
convert_to_box_coordinates_format_2(input_file, output_file)
    Convert a JSON file of detection data to a box coordinates format.

    Args:
    - input_file: str, the path to the input JSON file.
    - output_file: str, the path to the output JSON file.'''
def convert_to_box_coordinates_format_2(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    box_coordinates_format = {}
    for image_id, detections in data.items():
        boxes = [detection[0] + detection[1] for detection in detections]  # Extracting box coordinates
        box_coordinates_format[image_id] = boxes

    with open(output_file, 'w') as file:
        json.dump(box_coordinates_format, file, indent=4)


'''
calculate_iou(boxA, boxB)
    Calculate the intersection over union (IoU) of two bounding boxes.

    Args:
    - boxA: list, the coordinates of the first bounding box.
    - boxB: list, the coordinates of the second bounding box.

    Returns:
    - iou: float, the IoU of the two bounding boxes.'''
def calculate_iou(boxA, boxB):
    boxA = [int(coord) for coord in boxA]
    boxB = [int(coord) for coord in boxB]

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    if float(boxAArea + boxBArea - interArea) == 0:
        return 0
    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

'''
calculate_ap(preds_file, truths_file, threshold)
    Calculate the average precision (AP) for a set of predictions and ground truths.

    Args:
    - preds_file: str, the path to the JSON file containing the predictions.
    - truths_file: str, the path to the JSON file containing the ground truths.
    - threshold: float, the IoU threshold for matching predictions to ground truths.

    Returns:
    - ap: float, the average precision.'''

def calculate_ap(preds_file, truths_file, threshold):
    with open(preds_file, 'r') as file:
        preds_data = json.load(file)

    with open(truths_file, 'r') as file:
        truths_data = json.load(file)

    total_precisions = []
    total_recalls = []
    prcurve=[]
    for img_id in preds_data:
        preds = preds_data[img_id]
        truths = truths_data.get(img_id, [])

        # Convert ground truth coordinates to integer
        truths = [[int(coord) for coord in bbox] for bbox in truths]

        # Sort predictions by confidence score
        try:
            preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        except:
            print(preds)
            print("HELP")
    
        tp = 0  # True positives
        fp = 0  # False positives
        matched_truths = set()  # Track which ground truths have been matched
        precisions = []
        recalls = []
        
        # print(preds_sorted[0:5])
        for pred in preds_sorted[0]:
            # print(pred)
            pred_bbox = pred[0]  # Extract bounding box from prediction
            # print(pred_bbox)
            pred_bbox = [pred_bbox[0], pred_bbox[2], pred_bbox[1], pred_bbox[3]]
            best_iou = 0
            best_truth_index = None

            for i, truth_bbox in enumerate(truths):
                iou = calculate_iou(pred_bbox, truth_bbox)
                if iou > best_iou and i not in matched_truths:
                    best_iou = iou
                    best_truth_index = i

            if best_iou > threshold and best_truth_index is not None:
                matched_truths.add(best_truth_index)
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / len(truths) if len(truths) > 0 else 0
            prcurve.append((precision,recall))

        
    prcurve.sort(key=lambda x: x[1])

    # Use np.trapz to calculate the area under the curve
    precisions, recalls = zip(*prcurve)
    ap = np.trapz(precisions, x=recalls)

    return ap

'''
plotMAP()
    Plot the mean average precision (mAP) for a model over different IoU thresholds.
    
    Args:
    - None
    
    Returns:
    - None'''
def plotMAP():

    # run calculate ap over every threshold 0 to 1 in 0.05 increments and graph the results
    thresholds = [i/20 for i in range(21)]
    aps = [calculate_ap('../data/best_model_results_with_confidence.json', '../data/test_labels_boxes.json', threshold) for threshold in thresholds]

    import matplotlib.pyplot as plt

    plt.plot(thresholds, aps)
    plt.xlabel('Threshold')
    plt.ylabel('Average Precision')
    plt.title('Average Precision vs. Threshold (Best Model)')
    plt.show()

    #print average of aps
    print(np.mean(aps))


# calulate f1 score for the best model and the mean model and print the results
def calculate_f1_scores():
    best_model_f1 = calculate_batch_f1_score_from_json('../data/test_labels_only.json', '../data/best_model_labels_only.json')
    mean_model_f1 = calculate_batch_f1_score_from_json('../data/test_labels_only.json', '../data/mean_model_labels_only.json')
    print(f'Best Model F1 Score: {best_model_f1}')
    print(f'Mean Model F1 Score: {mean_model_f1}')

calculate_f1_scores()