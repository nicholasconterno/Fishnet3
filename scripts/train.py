import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_dataset import FishnetDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def train(model: torch.nn.Module, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          epochs: int = 100, 
          lr: float = 0.01,
          device: str = "cpu") -> torch.nn.Module:
    """
    Train the model using the given data loaders and hyperparameters.
    
    Args:
        model: torch.nn.Module, the model to train
        train_loader: DataLoader, the data loader for the training set
        val_loader: DataLoader, the data loader for the validation set
        epochs: int, the number of epochs
        lr: float, the learning rate
        device: str, the device to train on

    Returns:
        torch.nn.Module: the trained model
    """
    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    best_loss = float('inf')
    model_path = 'best_model.pth' 
    model.to(device)

    # Train the model
    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0

        # Training phase
        for inputs, targets in train_loader:
            inputs = list(img.to(device) for img in inputs)
            targets = [transform_targets_fixed(target) for target in targets]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Forward pass through the model
            loss_dict = model(inputs, targets)

            # The loss is returned by the model directly in training mode
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        # Validation phase
        val_loss = 0.0
        #model.eval()  DONT SET TO EVAL. THIS WILL STOP LOSS COMPUTATION. WEIRD THING FOR THIS PYTORCH MODEL
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = list(img.to(device) for img in inputs)
                targets = [transform_targets_fixed(target) for target in targets]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass through the model
                loss_dict = model(inputs, targets)

                # Sum up all the losses from the dictionary
                losses = sum(loss for loss in loss_dict.values())

                val_loss += losses.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    print(f"Training complete. Best validation loss: {best_loss:.4f}")
    # Load the best model weights
    model.load_state_dict(torch.load(model_path))
    return model


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



if __name__ == "__main__":
    # TODO:
    # Clean all of this up

    BATCH_SIZE = 32

    # Define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((400, 800))
    ])
    # Create the dataset
    dataset = FishnetDataset(labels_file="../data/labels.json", 
                            bucket_name="fishnet3-object-detection", 
                            gcp_cred_path="../fishnet3-56e06381ff35.json", 
                            transform=transform,
                            resize_shape=(400, 800)) 
    

    # Setup model and freeze the weights
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    for param in model.parameters():
        param.requires_grad = False

    # Change the final head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 26
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Create train, val, test splits
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Train the model
    trained_model = train(model, train_loader, val_loader, epochs=100, lr=0.001, device="cuda")




