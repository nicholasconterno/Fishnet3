from .faster_rcnn import process_image as process_image_rcnn
from .fishnet_detector import FishnetDetector
from .traditional_object_detection import process_image as process_image_rf
from . import IMG_FOLDER
import os

def object_detect(image_path):
    '''
    Process the input image with object detection using Faster R-CNN, FishNet and ML Approaches.
    Args:
        image_path: str, path to the input image file
        img_folder: str, path to the folder to save the processed images
    Returns:
        str: path to the processed image,
    '''
    # Process the image with Faster R-CNN
    faster_rcnn_pth = os.path.join(os.getcwd(), 'app/models/best_model.pth')
    rcnn_img_name = process_image_rcnn(image_path, faster_rcnn_pth)

    # Process the image with FishNet
    FishNet = FishnetDetector(model_path=faster_rcnn_pth)
    fishnet_img_name = f'fishnet_{os.path.basename(image_path)}'
    fishnet_img_path = os.path.join(IMG_FOLDER, fishnet_img_name)
    FishNet.detect(image_path, thresh=0.8, output_img_path=fishnet_img_path, show_labels=True)
    
    # Process the image with traditional object detection
    rf_model = os.path.join(os.getcwd(), 'app/models/randomforest_classifier32.pkl')
    rf_img_name = f'rf_{os.path.basename(image_path)}'
    rf_img_path = os.path.join(IMG_FOLDER, rf_img_name)
    process_image_rf(image_path, rf_model, output_path=rf_img_path, classifier_size=32, proba_threshold=0.99)
    
    return rcnn_img_name, fishnet_img_name, rf_img_name