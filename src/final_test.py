from models.DetachedHeadModel import DetachedHeadModel, compute_iou, torch_iou
from models.CustomDataset import CustomDataset
import clip
import torch
import matplotlib.pyplot as plt
from utils import xywh_to_topleft
from utils import categories

clip_model, preprocess = clip.load("RN50")

test_set = CustomDataset(
    split="test", model=clip_model, transform=preprocess, original_image_required=True
)

model = DetachedHeadModel()
checkpoint = torch.load("best-ever10.pt")
model.load_state_dict(checkpoint["model"])
# model.load_state_dict(torch.load("best-ever.pt", map_location=torch.device("cpu")))
model.eval()

cumulative_iou = 0.0
class_accuracy = 0.0
counter = 0
with torch.no_grad():
    for i in range(len(test_set)):
        image, text, bbox, category_id, orginal_image = test_set[i]
        counter += 1
        print("Counter: ", counter)
        predicted_bbox, unbound_class_probs = model(
            image.unsqueeze(0), text.unsqueeze(0)
        )
        predicted_class_id = torch.argmax(unbound_class_probs)
        class_name = categories[predicted_class_id.item()]
        predicted_bbox = predicted_bbox[0]
        width, height = orginal_image.size
        cumulative_iou += compute_iou(predicted_bbox, bbox)
        print("Intersection over union:", compute_iou(predicted_bbox, bbox))
        _, predicted_class_id = unbound_class_probs[0].max()

        # compute training accuracy
        class_accuracy += predicted_class_id.eq(category_id).item()
        # _, ax = plt.subplots(1)
        # ax.imshow(orginal_image)
        # bbox = xywh_to_topleft(bbox)
        # predicted_bbox = xywh_to_topleft(predicted_bbox.squeeze(0))
        # ax.annotate(
        #     class_name,
        #     (predicted_bbox[0] * width, predicted_bbox[1] * height),
        #     fontsize=12,
        #     fontweight="bold",
        #     color="red",
        #     ha="left",
        #     va="top",
        # )
        # ax.add_patch(
        #     plt.Rectangle(
        #         (bbox[0] * width, bbox[1] * height),
        #         bbox[2] * width,
        #         bbox[3] * height,
        #         fill=False,
        #         edgecolor="green",
        #         linewidth=2,
        #     )
        # )
        # ax.add_patch(
        #     plt.Rectangle(
        #         (predicted_bbox[0] * width, predicted_bbox[1] * height),
        #         predicted_bbox[2] * width,
        #         predicted_bbox[3] * height,
        #         fill=False,
        #         edgecolor="red",
        #         linewidth=2,
        #     )
        # )
        # plt.show()

    print("Average IOU: ", cumulative_iou / len(test_set))
