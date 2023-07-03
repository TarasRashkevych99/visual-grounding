from models.BestDetectorEver import BestDetectorEver
from models.CustomDataset import CustomDataset
import clip

clip_model, preprocess = clip.load("RN50")

detector = BestDetectorEver()

val_dataset = CustomDataset(split="val", shuffle=True, model=clip_model, transform=preprocess)

first_image, first_text, first_bbox, first_category_id = val_dataset[0]

print(first_image.shape)

detected_image = detector(first_image.unsqueeze(0))

print(detected_image.shape)