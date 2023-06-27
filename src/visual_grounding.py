from config import get_config
from models.BestDetectorEver import BestDetectorEver
from models.Metrics import Metrics
from ultralytics import YOLO
from clip import clip
from models.CustomDataset import CustomDataset
import utils
import torch


if __name__ == "__main__":
    device = get_config()["device"]

    yolo_model = YOLO("yolov8n.pt")

    clip_model, preprocess = clip.load("RN50")
    clip_model = clip_model.eval()

    train_dataset = CustomDataset(split="train", shuffle=True, model=clip_model, transform=preprocess)
    val_dataset = CustomDataset(split="val", shuffle=True)
    test_dataset = CustomDataset(split="test")

    metrics = Metrics(
        iou_threshold=0.5, prob_threshold=0.5, dataset_dim=len(val_dataset)
    )

    best_detector_ever = BestDetectorEver()

    print(train_dataset[0])
    train_loader = torch.utils.data.DataLoader(train_dataset, 1, shuffle=False)
    print(train_loader)
    iterator = iter(train_loader)
    for element in iterator:
        print(element)
        exit()
    

