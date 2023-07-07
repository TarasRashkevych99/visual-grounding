from models.BestDetectorEver import BestDetectorEver
from models.CustomDataset import CustomDataset
import clip
import torch
import matplotlib.pyplot as plt

clip_model, preprocess = clip.load("RN50")

test_set = CustomDataset(split="test", model=clip_model, transform=preprocess)

model = BestDetectorEver()
#model.load_state_dict(torch.load("best_detector_ever.pt"))
model.eval()

with torch.no_grad():
    for i in range(10):
        image, text, bbox, category_id = test_set[i]
        predicted_bbox = model(image.unsqueeze(0), text.unsqueeze(0))
        _, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))
        ax.add_patch(plt.Rectangle(bbox[:2]*224, bbox[2]*224, bbox[3]*224, fill=False, edgecolor='green', linewidth=2))
        ax.add_patch(plt.Rectangle(predicted_bbox[0][:2]*224, predicted_bbox[0][2]*224, predicted_bbox[0][3]*224, fill=False, edgecolor='red', linewidth=2))
        plt.show()