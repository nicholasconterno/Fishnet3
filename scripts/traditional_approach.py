import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from dataloading import load_data_json
import joblib
import matplotlib.pyplot as plt

# Define the class to id mapping
class_to_id = {'Human': 1, 'Swordfish': 2, 'Albacore': 3, 'Yellowfin tuna': 4, 'No fish': 5, 'Mahi mahi': 6, 'Skipjack tuna': 7, 'Unknown': 8, 'Wahoo': 9, 'Bigeye tuna': 10, 'Striped marlin': 11, 'Opah': 12, 'Blue marlin': 13, 'Escolar': 14, 'Shark': 15, 'Tuna': 16, 'Water': 17, 'Oilfish': 18, 'Pelagic stingray': 19, 'Marlin': 20, 'Great barracuda': 21, 'Shortbill spearfish': 22, 'Indo Pacific sailfish': 23, 'Lancetfish': 24, 'Long snouted lancetfish': 25, 'Black marlin': 26}
# Adjust this function to correctly map identifiers to file paths
def get_image_path(identifier):
    # Adjust the directory and file extension as needed
    return f'./data/foid_images_v100/images/{identifier}.jpg'

# Function to extract HOG features
def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)
    return hog_features

def build_features_and_labels(df, show_images=False, size=32):
    # Assuming df is loaded as shown in your printout
    features = []
    labels = []

    for index, row in df.iterrows():
        image_path = get_image_path(row['image'])  # Fetch the actual image path
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
            labels.append(class_to_id[class_label])

    features = np.array(features)
    labels = np.array(labels)
    return features, labels



if __name__ == '__main__':
    dict, label_df = load_data_json('data/labels.json')
    features, labels = build_features_and_labels(label_df, show_images=False, size=48)
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
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    clf.fit(X_train_scaled, y_train)

    # Save the classifier
    joblib.dump(clf, 'saved_models/randomforest_classifier32_entropy.pkl')
    print("Classifier saved to randomforest_classifier32_entropy.pkl")
    
    # Test the classifier
    y_pred = clf.predict(X_test_scaled)

    unique_labels_test = np.unique(y_test)
    unique_labels_pred = np.unique(y_pred)
    print("Unique labels in y_test:", unique_labels_test)
    print("Unique labels in y_pred:", unique_labels_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))