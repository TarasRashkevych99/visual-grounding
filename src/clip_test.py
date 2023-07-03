from typing import Any
from clip import clip
import torch
import torch.nn as nn
from models.CustomDataset import CustomDataset

class Classifier1(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model, _ = clip.load("RN50")
        visual_encoder = clip_model.visual
        layers = list(visual_encoder.children())
        self.model = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.model(x)

class Classifier2(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model, _ = clip.load("RN50")
        visual_encoder = clip_model.visual
        layers = list(visual_encoder.children())
        self.model = nn.Sequential(*layers[-1:])

    def forward(self, x):
        return self.model(x)
    
class Classifier3(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model, _ = clip.load("RN50")
        visual_encoder = clip_model.visual
        self.model = visual_encoder

    def forward(self, x):
        return self.model(x)

clip_model, preprocess = clip.load("RN50")
print(clip_model)
classifier_1 = Classifier1()
classifier_2 = Classifier2()
classifier_3 = Classifier3()

val_dataset = CustomDataset(split="val", shuffle=True, model=clip_model, transform=preprocess)
first_image, first_text, first_bbox, first_category_id = val_dataset[0]
print(first_image.shape)
classified_1 = classifier_1(first_image.unsqueeze(0))
print(classified_1.shape)
classified_2 = classifier_2(classified_1)
print(classified_2.shape)
print(classified_2)
classified_3 = classifier_3(first_image.unsqueeze(0))
print(classified_3.shape)
print(classified_3)

print(classifier_1)
