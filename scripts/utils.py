import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    """
    Custom two part loss function. 
    The first part is the cross-entropy loss for the class predictions.
    The second part is 1-IoU for the bounding box predictions.
    The two parts are combined with a weighted sum.

    Args:
        predictions:
        target: 
        phi: float, 0-1 weight to combine the two parts of the loss. 
            1 means only the bounding box loss is considered, 0 means only the class loss is considered.

    Returns:
        torch.Tensor: loss value
    """
    # Calculate the cross-entropy loss for the class predictions
    class_loss = torch.nn.functional.cross_entropy(predictions['class'], target['class'])
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
