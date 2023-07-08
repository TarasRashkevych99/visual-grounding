from models.DetachedHeadModel import DetachedHeadModel, compute_iou, torch_iou
from models.CustomDataset import CustomDataset
import clip
import torch
import matplotlib.pyplot as plt
from utils import xywh_to_topleft

clip_model, preprocess = clip.load("RN50")

test_set = CustomDataset(split="test", model=clip_model, transform=preprocess)

model = DetachedHeadModel()
#model.load_state_dict(torch.load("best-ever.pt", map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    for i in range(10):
        image, text, bbox, category_id = test_set[i]
        predicted_bbox = model(image.unsqueeze(0), text.unsqueeze(0))
        predicted_bbox = predicted_bbox[0]
        print("Predicted bbox:", predicted_bbox*224)
        print("Actual bbox:", bbox*224)
        print("Intersection over union:", compute_iou(predicted_bbox, bbox))
        print("IOU torchvision: ", torch_iou(predicted_bbox, bbox))
        _, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))
        print(bbox[:2]*224, bbox[2]*224, bbox[3]*224)
        bbox = xywh_to_topleft(bbox)
        predicted_bbox = xywh_to_topleft(predicted_bbox.squeeze(0))
        ax.add_patch(plt.Rectangle(bbox[:2]*224, bbox[2]*224, bbox[3]*224, fill=False, edgecolor='green', linewidth=2))
        ax.add_patch(plt.Rectangle(predicted_bbox[:2]*224, predicted_bbox[2]*224, predicted_bbox[3]*224, fill=False, edgecolor='red', linewidth=2))
        plt.show()