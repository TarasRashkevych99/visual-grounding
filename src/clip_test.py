from typing import Any
from clip import clip
import torch
import torch.nn as nn
from models.CustomDataset import CustomDataset

# _, preprocess = clip.load("RN50")

# visual_encoder = clip_model.visual

# layers = list(visual_encoder.children())
# print(layers)

# # Remove a specific layer
# # layer_to_remove = image_encoder.layer_name  # Replace with the name or index of the layer you want to remove
# # image_encoder_layers = list(image_encoder.children())
# image_encoder_layers = layers[:-1]
# # Create a new image encoder without the removed layer
# new_image_encoder = torch.nn.Sequential(*image_encoder_layers)
# layers = list(new_image_encoder.children())
# print(layers)

# # Update the CLIP model with the new image encoder
# clip_model.visual = new_image_encoder

# # Load the Dataset
# val_dataset = CustomDataset(split="val", shuffle=True, model=clip_model, transform=preprocess)
# first_image, first_text, first_bbox, first_category_id = val_dataset[0]
# print(first_image.shape)
# print(first_text)

# # Encode the image and the text
# first_image_embedding = clip_model.encode_image(first_image.unsqueeze(0))
# first_text_embedding = clip_model.encode_text(clip.tokenize(first_text))
# print(first_image_embedding.shape)

# # Define a new model with desired layers
# desired_layer_name = 'desired_layer_name'  # Replace with the name of the desired layer
# layers = list(image_encoder.children())
# desired_layer_index = [name for name, _ in image_encoder.named_children()].index(desired_layer_name)
# desired_layers = layers[:desired_layer_index + 1]
# new_model = torch.nn.Sequential(*desired_layers)

# # Forward pass through the new model
# input_tensor = torch.randn(1, 3, 224, 224)  # Replace with your input tensor
# desired_layer_output = new_model(input_tensor)

# # Use the desired layer output as needed
# print(desired_layer_output)

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
