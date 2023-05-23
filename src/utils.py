from DatasetSplit import DatasetSplit
from AnnotationSplitter import AnnotationSplitter


def get_partitions(transform=None):
    annotation_splitter = AnnotationSplitter()
    train = DatasetSplit(annotation_splitter.train_set_annotations.copy(), transform)
    val = DatasetSplit(annotation_splitter.val_set_annotations.copy(), transform)
    test = DatasetSplit(annotation_splitter.test_set_annotations.copy(), transform)
    return train, val, test
