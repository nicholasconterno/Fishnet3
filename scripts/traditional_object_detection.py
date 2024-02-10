import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from classify_boxes_rf import extract_hog_features, class_to_id

# Define the sliding window function
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Define the Non-Maximum Suppression function
def non_maximum_suppression(boxes, overlapThresh):
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
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    return boxes[pick].astype("int")

# Define the main processing function
def process_image(image_path, clf, window_size=(64, 64), step_size=16, show_images=True, nms_threshold=0.5, classifier_size=32, proba_threshold=0.99):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    detected_boxes = []
    for (x, y, window) in sliding_window(image, step_size=step_size, window_size=window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            print(f"Window size mismatch: {window.shape}")
            continue
        # resize to sizexsize for the classifier
        window_input = cv2.resize(window, (classifier_size, classifier_size))
        features = extract_hog_features(window_input).reshape(1, -1)
        pred = clf.predict(features)
        
        # if high probability, save the box
        proba = clf.predict_proba(features)[0]
        
        if np.max(proba) >= proba_threshold:
            print(f"Detected {pred},{list(class_to_id.keys())[pred[0]]} at {x}, {y} with probability {np.max(proba)}")
            # map pred to class name
            
            detected_boxes.append((x, y, window_size[0], window_size[1], pred[0], np.max(proba)))

    detected_boxes = np.array(detected_boxes)
    detected_objects = []
    if len(detected_boxes) > 0:
        # Apply NMS
        nms_boxes = non_maximum_suppression(detected_boxes[:, :4], nms_threshold)
        if show_images:
            plt.imshow(image, cmap='gray')
            for (x, y, w, h) in nms_boxes:
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none'))
            plt.show()
    
        for (x, y, w, h, label, proba) in detected_boxes:
            class_name = [key for key, value in class_to_id.items() if value == label][0]
            detected_objects.append({'x': x, 'y': y, 'w': w, 'h': h, 'label': class_name, 'probability': proba})

    if show_images and len(detected_boxes) > 0:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for (x, y, w, h, label, proba) in detected_boxes:
            # Draw rectangle
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none'))

            # Label with class name and probability
            class_name = [key for key, value in class_to_id.items() if value == label][0]
            label_text = f"{class_name}: {proba:.2f}"
            plt.text(x, y - 5, label_text, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        plt.show()

    return detected_objects

if __name__ == "__main__":
    # Load the classifier
    clf = joblib.load('saved_models/randomforest_classifier.pkl')
    # Have the user select an image to process
    process_image('data/test_images/dab2c170-db28-11ea-bc88-6fdfea10cd25.jpg', clf)
