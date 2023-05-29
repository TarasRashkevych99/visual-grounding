import os


def get_config():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = f"{project_root}/refcocog"
    annotations_path = f"{dataset_root}/annotations"
    refs_path = f"{annotations_path}/refs(umd).p"
    instances_path = f"{annotations_path}/instances.json"
    images_path = f"{dataset_root}/images"

    config = {
        "project_root": project_root,
        "dataset_root": dataset_root,
        "annotations_path": annotations_path,
        "refs_path": refs_path,
        "instances_path": instances_path,
        "images_path": images_path,
    }

    return config


if __name__ == "__main__":
    print(get_config())
