import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FishnetDetector:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_old = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
        self.model_old.to(self.device)
        self.model_old.eval()
        self.model_new = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
        in_features = self.model_new.roi_heads.box_predictor.cls_score.in_features
        num_classes = 26
        self.model_new.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model_new.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model_new.to(self.device)
        self.model_new.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def detect(self, 
               img_path: str, 
               thresh_human: float = 0.8, 
               thresh_fish: float = 0.5,
               output_img_path: str = None, 
               show_labels: bool = True):
        """
        Run inference on the image and return the output preditions. 
        Optionally, save the image with the bounding boxes drawn on it.

        Args:
        - img_path: str, the path to the image to run inference on.
        - thresh_human: float, the score threshold for human detection.
        - thresh_fish: float, the score threshold for fish detection.
        - output_img_path: str, the path to save the annotated image.
        - show_labels: bool, whether to display labels on the image.
        """
        img = self._prepare_img(img_path)
        with torch.no_grad():
            # Make predictions with both models
            outputs_old = self.model_old(img)
            outputs_new = self.model_new(img)
            # Filter outputs by threshold
            outputs_old = self._filter_outputs_by_threshold(outputs_old, thresh_human)
            outputs_new = self._filter_outputs_by_threshold(outputs_new, thresh_fish)
            # Combine outputs
            combined_outputs = self._combine_outputs(outputs_old, outputs_new)
            # Display bounding boxes
            if output_img_path:
                self._display_bounding_boxes(img, combined_outputs, min(thresh_human, thresh_fish), show_labels, output_img_path)
            else:
                pass
                # self._display_bounding_boxes(img, combined_outputs, min(thresh_human, thresh_fish), show_labels)
        return combined_outputs
    

    def _prepare_img(self, img_path: str):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).to(self.device).unsqueeze(0)
        return img 
    

    def _filter_outputs_by_threshold(self, outputs: list, thresh: float) -> dict:
        filtered_outputs = []
        for output in outputs:
            # Apply threshold to 'scores' and filter 'boxes' and 'labels' accordingly
            keep = output['scores'] > thresh
            filtered_output = {key: value[keep] for key, value in output.items()}
            filtered_outputs.append(filtered_output)
        return filtered_outputs[0]
    

    def _convert_labels_to_string_old_model(self, outputs: dict) -> dict:
        converted_outputs = {"boxes": [], "labels": [], "scores": []}
        # Convert numerical labels to string labels
        for i in range(len(outputs["scores"])):
            if outputs["labels"][i] == 1:
                converted_outputs["labels"].append("Human")
                converted_outputs["boxes"].append(outputs["boxes"][i])
                converted_outputs["scores"].append(outputs["scores"][i])
        return converted_outputs


    def _convert_labels_to_string_new_model(self, outputs: dict) -> dict:
        id_to_class = {0: "Human", 1: "Swordfish", 2: "Albacore", 3: "Yellowfin tuna", 4: "No fish", 5: "Mahi mahi", 6: "Skipjack tuna", 7: "Unknown", 8: "Wahoo", 9: "Bigeye tuna", 10: "Striped marlin", 11: "Opah", 12: "Blue marlin", 13: "Escolar", 14: "Shark", 15: "Tuna", 16: "Water", 17: "Oilfish", 18: "Pelagic stingray", 19: "Marlin", 20: "Great barracuda", 21: "Shortbill spearfish", 22: "Indo Pacific sailfish", 23: "Lancetfish", 24: "Long snouted lancetfish", 25: "Black marlin"}
        converted_outputs = {"boxes": [], "labels": [], "scores": []}
        # Convert numerical labels to string labels
        for i in range(len(outputs["scores"])):
            converted_outputs["labels"].append(id_to_class[outputs["labels"][i].item()])
            converted_outputs["boxes"].append(outputs["boxes"][i])
            converted_outputs["scores"].append(outputs["scores"][i])
        return converted_outputs
    

    def _combine_outputs(self, outputs_old: dict, outputs_new: dict):
        # Convert numerical labels to string labels
        outputs_old = self._convert_labels_to_string_old_model(outputs_old)
        outputs_new = self._convert_labels_to_string_new_model(outputs_new)
        combined_outputs = {}
        combined_outputs["boxes"] = outputs_old["boxes"] + outputs_new["boxes"]
        combined_outputs["labels"] = outputs_old["labels"] + outputs_new["labels"]
        combined_outputs["scores"] = outputs_old["scores"] + outputs_new["scores"]
        return [combined_outputs]


    def _display_bounding_boxes(self, 
                                input_image: torch.Tensor, 
                                model_outputs: dict, 
                                thresh: float = 0.8, 
                                show_labels: bool = False, 
                                save_path = None):
        """
        Display bounding boxes on the image, annotate each with its label and score, and save the annotated image.

        Args:
        - input_image: torch.Tensor, the input image tensor.
        - model_outputs: dict, the output from the detection model.
        - thresh: float, the score threshold for displaying bounding boxes.
        - show_labels: bool, whether to display labels on the image.
        - save_path: str, the path to save the annotated image.
        """

        # Assuming 'labels' and 'scores' are keys in the model_outputs dict
        boxes = model_outputs[0]['boxes']
        scores = model_outputs[0]['scores']
        labels = model_outputs[0]['labels']  # Assuming there's a 'labels' key

        # Convert to numpy arrays and filter by score threshold
        cleaned_boxes = []
        cleaned_scores = []
        cleaned_labels = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= thresh:
                box = box.detach().cpu().numpy().astype(int)
                cleaned_boxes.append(box)
                cleaned_scores.append(score.detach().cpu().numpy())
                cleaned_labels.append(str(label))

        # Create figure and axes
        fig, ax = plt.subplots()
        # Display the image
        ax.imshow(input_image[0].permute(1, 2, 0).cpu().numpy(), interpolation='nearest')  # Adjusted for tensor format (C, H, W)

        # Create a Rectangle patch and label for each cleaned box and add to the plot
        for box, score, label in zip(cleaned_boxes, cleaned_scores, cleaned_labels):
            # Draw rectangle
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if show_labels:
                # Annotate label and score
                ax.text(box[0], box[1]-10, f'{label}: {score*100:.0f}%', color='#8B0000', fontsize=8, fontweight='bold')

        # Hide axes
        ax.axis('off')
        # Save the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=500)
        else:
            plt.show()



if __name__ == "__main__":
    # Create a detector using saved weights
    detector = FishnetDetector(model_path="../data/best_model.pth")
    # Run inference on an image
    output = detector.detect(img_path="../data/test_image.jpg", 
                             thresh_human=0.8,
                             thresh_fish=0.6,
                             output_img_path="../data/test_image_annotated.jpg",
                             show_labels=True)
    

    # Print output in a readable format
    for i in range(len(output[0]["labels"])):
        print(f"Label: {output[0]['labels'][i]}, Score: {output[0]['scores'][i]*100:.0f}%, Bounding Box: {output[0]['boxes'][i]}")

    