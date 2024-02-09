import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def calc_iou(box1: list, box2: list) -> float:
    """
    Calculate the intersection over union (IoU) of two bounding boxes.

    Args:
        box1: list of floats (x_min, y_min, x_max, y_max)
        box2: list of floats (x_min, y_min, x_max, y_max)

    Returns:
        float: intersection over union (IoU) between box1 and box2
    """
    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # compute the IoU
    return intersection_area / union_area


def custom_loss(predictions: torch.Tensor, target: torch.Tensor, phi: float = 0.5) -> torch.Tensor:
    # Ensure predictions['class'] and target['class'] are 1D tensors
    class_pred = predictions['class'].unsqueeze(0) if predictions['class'].ndim == 0 else predictions['class']
    class_target = target['class'].unsqueeze(0) if target['class'].ndim == 0 else target['class']

    # Calculate the cross-entropy loss for the class predictions
    class_loss = torch.nn.functional.cross_entropy(class_pred, class_target)
    
    # Calculate the 1-IoU for the bounding box predictions
    iou_loss = 1 - calc_iou(predictions['box'], target['box'])
    
    # Combine the two parts with a weighted sum
    return phi * class_loss + (1 - phi) * iou_loss



def display_bounding_boxes(input_image: torch.Tensor, model_outputs: dict, thresh: float = 0.8) -> None:
    """
    Display the input image with the bounding boxes of the detected objects.

    Args:
        input_image: torch.Tensor of shape (3, H, W)
        model_outputs: dict with the model outputs
        thresh: float, threshold for the confidence score
    """
    boxes = model_outputs[0]['boxes']
    cleaned_boxes = []
    for box in boxes:
        box = box.detach().numpy()
        box = box.astype(int)
        cleaned_boxes.append(box)

    scores = model_outputs[0]['scores']
    for i in range(len(scores)):
        if scores[i] < thresh:
            cleaned_boxes[i] = None
            
    cleaned_boxes = [box for box in cleaned_boxes if box is not None]

    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(input_image[0].permute(1, 2, 0))

    # Create a Rectangle patch for each cleaned box and add to the plot
    for box in cleaned_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def calculate_iou_matrix(pred_boxes, true_boxes):
    """
    Calculate the IoU matrix between each pair of predicted and true boxes.

    Args:
        pred_boxes (tensor): Predicted bounding boxes, shape (num_pred, 4).
        true_boxes (tensor): True bounding boxes, shape (num_true, 4).

    Returns:
        Tensor: Matrix of IoU values, shape (num_pred, num_true).
    """
    num_pred = pred_boxes.shape[0]
    num_true = true_boxes.shape[0]

    # Initialize matrix to store IoU values
    iou_matrix = torch.zeros((num_pred, num_true))

    # Calculate IoU for each pair of boxes
    for i in range(num_pred):
        for j in range(num_true):
            iou_matrix[i, j] = calc_iou(pred_boxes[i], true_boxes[j])

    return iou_matrix

def find_matches(iou_matrix, iou_threshold=0.1):
    """
    Find the matches between predicted and true boxes based on the IoU matrix.

    Args:
        iou_matrix (Tensor): Matrix of IoU values, shape (num_pred, num_true).
        iou_threshold (float): Threshold for IoU to consider a match.

    Returns:
        List[Tuple[int, int]]: List of matches in the format (pred_idx, true_idx).
    """
    matches = []

    # Convert the IoU matrix to numpy for easier processing
    iou_matrix_np = iou_matrix.numpy()

    while True:
        # Find the index of the maximum IoU value
        pred_idx, true_idx = np.unravel_index(np.argmax(iou_matrix_np, axis=None), iou_matrix_np.shape)

        # Get the max IoU value
        max_iou = iou_matrix_np[pred_idx, true_idx]

        # Check if the max IoU is above the threshold
        if max_iou < iou_threshold:
            break

        # Add the match
        matches.append((pred_idx, true_idx))

        # Set the entire row and column to zero to avoid duplicate matches
        iou_matrix_np[pred_idx, :] = 0
        iou_matrix_np[:, true_idx] = 0

        # If all true boxes or predicted boxes have been matched, exit the loop
        if np.all(iou_matrix_np == 0):
            break
    #return matches with corresponding IoU
    print(matches)
    return matches





def calculate_total_loss_for_image(predictions, targets, phi=0.5, unmatched_penalty=1.0, iou_threshold=0.1):
    """
    Calculate the total loss for an image, including the loss for matched boxes and a penalty for unmatched boxes.

    Args:
        predictions: dict, containing predicted boxes and classes
        targets: dict, containing true boxes and classes
        phi: float, weight factor for combining class loss and bounding box loss
        unmatched_penalty: float, penalty to add for each unmatched box
        iou_threshold: float, threshold for considering a box match

    Returns:
        torch.Tensor: total loss for the image
    """
    pred_boxes = predictions['boxes']
    true_boxes = targets['boxes']
    pred_classes = predictions['classes']
    true_classes = targets['classes']

    # Calculate IoU matrix and find matches
    iou_matrix = calculate_iou_matrix(pred_boxes, true_boxes)
    matches = find_matches(iou_matrix, iou_threshold)

    total_loss = 0.0
    matched_pred_ind = set()
    matched_true_ind = set()
    # Calculate loss for matched boxes
    for pred_idx, true_idx in matches:
        matched_pred = {'class': pred_classes[pred_idx], 'box': pred_boxes[pred_idx]}
        matched_true = {'class': true_classes[true_idx], 'box': true_boxes[true_idx]}
        total_loss += custom_loss(matched_pred, matched_true, phi)
        matched_pred_ind.add(pred_idx)
        matched_true_ind.add(true_idx)

    # Add penalty for unmatched boxes
    
    unmatched_pred_count = len(pred_boxes) - len(matched_pred_ind)
    unmatched_true_count = len(true_boxes) - len(matched_true_ind)

    total_loss += unmatched_penalty * (unmatched_pred_count + unmatched_true_count)

    return total_loss


def transform_targets_fixed(raw_targets):
    # Define a mapping from class names to class IDs (assuming 'Human' is class 1)
    class_to_id = {'Human': 0, 'Swordfish': 1, 'Albacore': 2, 'Yellowfin tuna': 3, 'No fish': 4, 'Mahi mahi': 5, 'Skipjack tuna': 6, 'Unknown': 7, 'Wahoo': 8, 'Bigeye tuna': 9, 'Striped marlin': 10, 'Opah': 11, 'Blue marlin': 12, 'Escolar': 13, 'Shark': 14, 'Tuna': 15, 'Water': 16, 'Oilfish': 17, 'Pelagic stingray': 18, 'Marlin': 19, 'Great barracuda': 20, 'Shortbill spearfish': 21, 'Indo Pacific sailfish': 22, 'Lancetfish': 23, 'Long snouted lancetfish': 24, 'Black marlin': 25}

    # Initialize lists to hold transformed targets
    boxes = []
    labels = []

    for item in raw_targets:
        # Check if the target is not 'Missing'
        if item[2] != 'Missing':
            # Convert coordinate strings to float and correctly order as (xmin, ymin, xmax, ymax)
            box = [float(item[0][0]), float(item[1][0]), float(item[0][1]), float(item[1][1])]
            boxes.append(box)
            # Convert class name to class ID and add to labels list
            labels.append(class_to_id[item[2]])

    # Convert lists to PyTorch tensors
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    # Create the target dictionary
    target_dict = {'boxes': boxes_tensor, 'labels': labels_tensor}

    return target_dict


# import torch

# # Mock data
# # Predicted boxes (format: [x_min, y_min, x_max, y_max])
# pred_boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32)  # One box overlaps significantly with a true box, the other doesn't

# # True boxes
# true_boxes = torch.tensor([[20, 20, 40, 40], [40, 40, 300, 300]], dtype=torch.float32)  # One box overlaps with a pred box, the other is far away

# # Predicted classes (let's assume a binary classification: 0 and 1)
# pred_classes = torch.tensor([0.0, 1.0])  # Let's assume the first prediction is correct, and the second is incorrect

# # True classes
# true_classes = torch.tensor([0.0, 0.0])  # Both true boxes belong to class 0

# predictions = {'boxes': pred_boxes, 'classes': pred_classes}
# targets = {'boxes': true_boxes, 'classes': true_classes}

# # Assuming the custom_loss function and other dependencies are defined correctly
# # Calculate the total loss
# total_loss = calculate_total_loss_for_image(predictions, targets, phi=0.5, unmatched_penalty=1.0, iou_threshold=0.1)
# print(total_loss)