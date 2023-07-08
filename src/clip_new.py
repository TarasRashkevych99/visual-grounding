from models.DetachedHeadModel import DetachedHeadModel
from models.FullHeadModel import FullHeadModel
from models.CustomDataset import CustomDataset
import clip

clip_model, preprocess = clip.load("RN50")

detector = FullHeadModel()

detector.eval()

val_dataset = CustomDataset(split="val", model=clip_model, transform=preprocess)

first_image, first_text, first_bbox, first_category_id = val_dataset[0]

print(first_image.shape)

detected_image = detector(first_image.unsqueeze(0), first_text.unsqueeze(0))

print(detected_image.shape)