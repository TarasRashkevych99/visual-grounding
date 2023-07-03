from PIL import Image
import torch
from torch.utils.data import Dataset
from config import get_config
import json
import numpy as np
import random
import utils
import clip


class CustomDataset(Dataset):
    def __init__(self, transform=None, split="train", shuffle=False, model=None):
        super().__init__()
        self.clip = model
        self.annotations_path = get_config()["annotations_path"]
        if split == "train":
            self.instances = list(
                json.load(
                    open(f"{self.annotations_path}/instances_train.json")
                ).values()
            )
        elif split == "val":
            self.instances = list(
                json.load(open(f"{self.annotations_path}/instances_val.json")).values()
            )
        elif split == "test":
            self.instances = list(
                json.load(open(f"{self.annotations_path}/instances_test.json")).values()
            )
        self.images_path = get_config()["images_path"]
        self.transform = transform
        if shuffle:
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        image = Image.open(f"{self.images_path}/{self.instances[index]['image_name']}")
        bbox = torch.tensor(self.instances[index]["bbox"])
        if self.transform:
            image = self.transform(image).to(get_config()["device"])
        sentences = self.instances[index+10]["sentences"]
        idx = np.random.randint(len(sentences))
        random_sentence = sentences[idx]
        #random_sentence = clip.tokenize(random_sentence)).to(get_config()["device"])
        category_id = self.instances[index]["category_id"]
        embedding = utils.encode_data_with_clip(self.clip, image, random_sentence)
        return image, random_sentence, bbox, category_id
