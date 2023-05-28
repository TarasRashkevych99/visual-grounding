from PIL import Image
from torch.utils.data import Dataset
from config import get_config
import json
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self, transform=None, split="train", shuffle=False):
        super().__init__()
        self.annotations_path = get_config()["annotations_path"]
        if split == "train":
            self.instances = list(json.load(open(f"{self.annotations_path}/instances_train.json")).values())
        elif split == "val":
            self.instances = list(json.load(open(f"{self.annotations_path}/instances_val.json")).values())
        elif split == "test":
            self.instances = list(json.load(open(f"{self.annotations_path}/instances_test.json")).values())
        self.image_path = get_config()["images_path"]
        self.transform = transform
        if shuffle:
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        image = Image.open("./refcocog/images/" + self.instances[index]["image_name"])
        bbox = self.instances[index]["bbox"]
        if self.transform:
            image = self.transform(image)
        sentences = self.instances[index]["sentences"]
        idx = np.random.randint(len(sentences))
        random_sentence = sentences[idx]
        category_id = self.instances[index]["category_id"]
        return image, random_sentence, bbox, category_id