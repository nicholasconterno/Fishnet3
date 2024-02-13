import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog

# Original class to ID mapping
class_to_id = {'Human': 0, 'Swordfish': 1, 'Albacore': 2, 'Yellowfin tuna': 3, 'No fish': 4, 'Mahi mahi': 5, 'Skipjack tuna': 6, 'Unknown': 7, 'Wahoo': 8, 'Bigeye tuna': 9, 'Striped marlin': 10, 'Opah': 11, 'Blue marlin': 12, 'Escolar': 13, 'Shark': 14, 'Tuna': 15, 'Water': 16, 'Oilfish': 17, 'Pelagic stingray': 18, 'Marlin': 19, 'Great barracuda': 20, 'Shortbill spearfish': 21, 'Indo Pacific sailfish': 22, 'Lancetfish': 23, 'Long snouted lancetfish': 24, 'Black marlin': 25}
# Create inverse mapping from ID to class name
id_to_class = {v: k for k, v in class_to_id.items()}

def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    '''
    Extract HOG features from the input image.
    Args:
        image: np.array, input image
        pixels_per_cell: tuple, size of each cell in pixels
        cells_per_block: tuple, number of cells in each block
        orientations: int, number of orientation bins
    Returns:
        np.array: HOG features
    '''
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)
    return hog_features

def sliding_window(image, step_size, window_size):
    '''
    Slide a window across the image.
    Args:
        image: np.array, input image
        step_size: int, step size for the sliding window
        window_size: tuple, size of the sliding window
    Yields:
        tuple: (x, y, window) coordinates and image window
    '''
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def non_maximum_suppression(boxes, thresh):
    '''
    Apply non-maximum suppression to the bounding boxes.
    Args:
        boxes: np.array, bounding boxes
        thresh: float, threshold for overlapping boxes
    Returns:
        np.array: bounding boxes after non-maximum suppression
    '''
    if len(boxes) == 0:
        return []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2] + x1
    y2 = boxes[:,3] + y1
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        # Add the index value to the list of picked indexes
        pick.append(i)
        
        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > thresh)[0])))
        
    return boxes[pick]

def process_image(image_path, rf_path, window_size=(128, 128), step_size=16, nms_threshold=0.5, classifier_size=32, proba_threshold=0.99, output_path=None):
    '''
    Process the input image using the random forest classifier and sliding window.
    Create bounding boxes around detected objects and apply non-maximum suppression.
    Args:
        image_path: str, path to the input image file
        rf_path: str, path to the random forest classifier
        window_size: tuple, size of the sliding window
        step_size: int, step size for the sliding window
        nms_threshold: float, threshold for non-maximum suppression
        classifier_size: int, size to resize the images (square)
        proba_threshold: float, threshold for the classifier probability
        output_path: str, path to save the annotated image
    Returns:
        list: detected objects
    '''
    # Load the classifier
    clf = joblib.load(rf_path)

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    detected_boxes = []
    # Slide the window across the image
    for (x, y, window) in sliding_window(image, step_size=step_size, window_size=window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            print(f"Window size mismatch: {window.shape}")
            continue
        # resize to sizexsize for the classifier
        window_input = cv2.resize(window, (classifier_size, classifier_size))
        features = extract_hog_features(window_input).reshape(1, -1)
        pred = clf.predict(features)
        # get probability
        proba = clf.predict_proba(features)[0]

        # if the probability is high enough, save the box
        if np.max(proba) >= proba_threshold:
            # print(f"Detected,{list(class_to_id.keys())[pred[0]]} at {x}, {y} with probability {np.max(proba)}")
            detected_boxes.append((x, y, window_size[0], window_size[1], pred[0], np.max(proba)))
        
    detected_boxes = np.array(detected_boxes)
    if output_path:
        # Draw the boxes on the image and save it
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        if len(detected_boxes) > 0:
            # Apply NMS
            nms_boxes = non_maximum_suppression(detected_boxes, nms_threshold)

            # Draw the boxes
            for (x, y, w, h, lbel, proba) in nms_boxes:
                ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none'))
                # Label with class name and probability
                class_name = id_to_class[lbel]
                label_text = f"{class_name}: {proba:.2f}"
                ax.text(x, y - 5, label_text, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        
        fig.tight_layout()
        # remove axis
        ax.axis('off')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=500)
        print(f"Annotated image saved to {output_path}")

    return 1
