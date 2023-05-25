import pickle
from config import get_config
import random


class AnnotationSplitter:
    def __init__(self):
        super().__init__()
        self.train_set_annotations = []
        self.val_set_annotations = []
        self.test_set_annotations = []
        self.annotations_path = get_config()["annotations_path"]
        annotations = pickle.load(open(self.annotations_path, "rb"))
        for a in annotations:
            if a["split"] == "train":
                self.train_set_annotations.append(a)
            if a["split"] == "val":
                self.val_set_annotations.append(a)
            if a["split"] == "test":
                self.test_set_annotations.append(a)
        random.shuffle(self.val_set_annotations)
        random.shuffle(self.test_set_annotations)
        random.shuffle(self.train_set_annotations)
