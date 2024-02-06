import pandas as pd 
from sklearn.model_selection import train_test_split


def load_csv(file_path):
    return pd.read_csv(file_path)

def get_unique_classes(data):
    '''
    Get unique classes and count of each class in the data
    '''
    # get unique classes and count
    unique_classes = data['label_l1'].value_counts()
    # drop ones with less than 50 instances
    # unique_classes = unique_classes[unique_classes > 50]
    # count how many classes are left
    print(f"Unique classes: {len(unique_classes)}")
    return unique_classes

def train_test_val_split(data, test_size=0.2):
    '''
    split the data into train, test, and validation sets stratified by the label
    '''
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data['label_l1'])
    train_data, val_data = train_test_split(train_data, test_size=test_size, stratify=train_data['label_l1'])
    return train_data, test_data, val_data

if __name__ == '__main__':
    data = load_csv('data/fishnet_labels.csv')
    print(get_unique_classes(data))
    print(data.head())
    train_data, test_data, val_data = train_test_val_split(data)
    # print shapes of the data
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Validation data shape: {val_data.shape}")