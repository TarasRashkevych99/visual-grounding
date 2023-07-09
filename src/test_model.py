from models.DetachedHeadModel import DetachedHeadModel, compute_iou, torch_iou
from models.FullHeadModel import FullHeadModel
from models.CustomDataset import CustomDataset
import clip
import torch
import matplotlib.pyplot as plt
from utils import xywh_to_topleft
from utils import categories

clip_model, preprocess = clip.load("RN50")

test_set = CustomDataset(split="test", model=clip_model, transform=preprocess)

model = DetachedHeadModel()
model.load_state_dict(torch.load("real-model.pt", map_location=torch.device("cpu")))
model.eval()

with torch.no_grad():
    for i in range(100, 150):
        image, text, bbox, category_id = test_set[i]
        predicted_bbox, unbound_class_probs = model(
            image.unsqueeze(0), text.unsqueeze(0)
        )
        predicted_class_id = torch.argmax(unbound_class_probs)
        class_name = categories[predicted_class_id]
        predicted_bbox = predicted_bbox[0]
        width, height = image.shape[1:]
        print("Intersection over union:", compute_iou(predicted_bbox, bbox))
        print("IOU torchvision: ", torch_iou(predicted_bbox, bbox))
        _, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))
        bbox = xywh_to_topleft(bbox)
        predicted_bbox = xywh_to_topleft(predicted_bbox.squeeze(0))
        ax.annotate(
            class_name,
            (predicted_bbox[0], predicted_bbox[1]),
            fontsize=8,
            fontweight="bold",
            color="white",
            ha="left",
            va="top",
        )
        ax.add_patch(
            plt.Rectangle(
                (bbox[0] * width, bbox[1] * height),
                bbox[2] * width,
                bbox[3] * height,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (predicted_bbox[0] * width, predicted_bbox[1] * height),
                predicted_bbox[2] * width,
                predicted_bbox[3] * height,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
        plt.show()
