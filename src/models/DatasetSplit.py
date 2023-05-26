from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
from config import get_config


class DatasetSplit(Dataset):
    def __init__(self, annotations, transform=None):
        super().__init__()
        self.annotations = annotations
        self.image_path = get_config()["images_path"]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image_name = f"{self.image_path}/{annotation['file_name'].replace('_' + str(annotation['ann_id']), '')}"
        annotation_id = annotation["ann_id"]
        image = Image.open(image_name)
        if self.transform:
            image = self.transform(image)
        sentences = annotation["sentences"]
        idx = np.random.randint(len(sentences))
        random_sentence = sentences[idx]["raw"]
        return image, random_sentence, image_name, annotation_id
