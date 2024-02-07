import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import joblib

# Function to extract features from an image
def extract_features(image_path):
    # Here, we're using a very basic feature: the raw pixel values (flattened)
    # In practice, you'd want to use more sophisticated features
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image_resized = cv2.resize(image, (32, 32), interpolation= cv2.INTER_LANCZOS4)  # Resize to make the feature vector size consistent
    features = image_resized.flatten()
    return features

# Load your dataset
df = pd.read_csv('path_to_your_dataset.csv')  # Update this to the path of your dataset

# Extract features for each image
features = []
for index, row in df.iterrows():
    feature = extract_features(row['imagepath'])
    if feature is not None:
        features.append(feature)
    else:
        df.drop(index, inplace=True)  # Remove rows with images that couldn't be processed

# Assuming all images were processed successfully
X = np.array(features)
y = df[['min_x', 'min_y', 'max_x', 'max_y']].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and train the Random Forest regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predict bounding boxes for the test set
y_pred = regressor.predict(X_test)

# Evaluate the regressor here (e.g., using mean squared error, mean absolute error, etc.)

print("Model trained and bounding boxes predicted.")
# Save the trained model

joblib.dump(regressor, 'models/random_forest_regressor.pkl')
print("Model saved as 'random_forest_regressor.pkl'.")

