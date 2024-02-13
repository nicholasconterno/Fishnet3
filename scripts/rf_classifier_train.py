import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from dataloading import load_data_json
import joblib
import matplotlib.pyplot as plt

# Define the class to id mapping
CLASS_TO_ID = {'Human': 0, 'Swordfish': 1, 'Albacore': 2, 'Yellowfin tuna': 3, 'No fish': 4, 'Mahi mahi': 5, 'Skipjack tuna': 6, 'Unknown': 7, 'Wahoo': 8, 'Bigeye tuna': 9, 'Striped marlin': 10, 'Opah': 11, 'Blue marlin': 12, 'Escolar': 13, 'Shark': 14, 'Tuna': 15, 'Water': 16, 'Oilfish': 17, 'Pelagic stingray': 18, 'Marlin': 19, 'Great barracuda': 20, 'Shortbill spearfish': 21, 'Indo Pacific sailfish': 22, 'Lancetfish': 23, 'Long snouted lancetfish': 24, 'Black marlin': 25}

def get_image_path(identifier):
    '''
    Get the image path from the identifier. Adjust to fix the path to the images.
    Args:
        identifier: str, image identifier
    Returns:
        str: path to the image
    '''
    return f'./data/foid_images_v100/images/{identifier}.jpg'

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

def build_features_and_labels(df, show_images=False, size=32):
    '''
    Build Features and Labels for training the random forest classifier. 
    Extracting HOG features from each bounding box region of the images.
    Args:
        df: DataFrame, data with names and bounding boxes
        show_images: bool, whether to show the images and their segmented bounding boxes
        size: int, size to resize the images (square)
    '''
    features = []
    labels = []

    for _, row in df.iterrows():
        image_path = get_image_path(row['image'])  # Fetch the actual image path from the image name
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        for bbox_data in row['label']:
            (x1, x2), (y1, y2), class_label = bbox_data[0], bbox_data[1], bbox_data[2]
            
            # correctly order as x1 = xmin, y1 = ymin, x2 = xmax, y2 = ymax
            xmin, xmax = min(int(x1), int(x2)), max(int(x1), int(x2))
            ymin, ymax = min(int(y1), int(y2)), max(int(y1), int(y2))

            # if any value is negative, round to 0
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            if show_images:
                print(f"Processing image: {image_path}, class: {class_label}, bbox: {xmin},{ymin} - {xmax},{ymax}")
                print(f"Image shape: {image.shape}")

                # display the image with the bounding box
                plt.imshow(image)
                plt.title('Original Image')
                plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='r', linewidth=2))
                plt.show()

            # get target box
            cropped_image = image[ymin:ymax, xmin:xmax]

            if show_images:
                # display the cropped image
                plt.imshow(cropped_image)
                plt.title('Cropped Image')
                plt.show()

            try: 
                resized_image = cv2.resize(cropped_image, (size, size))  # Resize for a consistent feature vector size
            except cv2.error as e:
                print(f"Error resizing image: {image_path}, class: {class_label}, bbox: {xmin},{ymin} - {xmax},{ymax}")
                continue
            # show the resized image
            if show_images:
                plt.imshow(resized_image)
                plt.title('Resized Image')
                plt.show()
            hog_features = extract_hog_features(resized_image)
            features.append(hog_features)
            labels.append(CLASS_TO_ID[class_label])

    features = np.array(features)
    labels = np.array(labels)
    return features, labels



if __name__ == '__main__':
    # Load the data
    dict, label_df = load_data_json('data/labels.json')
    features, labels = build_features_and_labels(label_df, show_images=False, size=32)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print("Training data shape:", X_train.shape, y_train.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training the classifier...")
    # Train the RandomForestClassifier for multi-class
    # clf = RandomForestClassifier(n_estimators=100, criterion='gini')
    clf = BalancedRandomForestClassifier(n_estimators=50)
    clf.fit(X_train_scaled, y_train)

    # Save the classifier
    file_name = 'saved_models/balancedrandomforest_50_classifier32.pkl'
    joblib.dump(clf, file_name)
    print(f"Classifier saved to {file_name}.")
    
    # Test the classifier
    y_pred = clf.predict(X_test_scaled)

    unique_labels_test = np.unique(y_test)
    unique_labels_pred = np.unique(y_pred)
    print("Unique labels in y_test:", unique_labels_test)
    print("Unique labels in y_pred:", unique_labels_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))