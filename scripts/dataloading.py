import torch
import json

def load_data_json(file_path):
    """
    Load the JSON file from the given file path.
    TODO: Upadate to fit the actual data format
    Args:
        file_path: str, path to the JSON file

    Returns:
        dict: data from the JSON file
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_dataloaders(data):
    """
    Create the dataloaders for the training, validation, and test sets.

    Args:
        data: dict, data from the JSON file

    Returns:
        dict: dataloaders for the training, validation, and test sets
    """
    # split data in to train, val, and test using scikit-learn
    # TODO: Implement the split

    # Create the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=32, shuffle=True, num_workers=2),
        'val': torch.utils.data.DataLoader(data['val'], batch_size=32, shuffle=False, num_workers=2),
        'test': torch.utils.data.DataLoader(data['test'], batch_size=32, shuffle=False, num_workers=2)
    }
    return dataloaders