import os


def get_config():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, "refcocog")
    annotations_path = os.path.join(dataset_root, "annotations/refs(umd).p")
    images_path = os.path.join(dataset_root, "images/")

    config = {
        "project_root": project_root,
        "dataset_root": dataset_root,
        "annotations_path": annotations_path,
        "images_path": images_path,
    }

    return config
