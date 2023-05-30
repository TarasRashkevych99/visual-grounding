import json
import pickle

from config import get_config

new_dataset_train = {}
new_dataset_test = {}
new_dataset_val = {}


instances = json.load(open(get_config()["instances_path"], "r"))
refs = pickle.load(open(get_config()["refs_path"], "rb"))

ids_train = set()
ids_test = set()
ids_val = set()

for ref in refs:
    if ref["split"] == "train":
        ids_train.add(ref["ann_id"])
    elif ref["split"] == "val":
        ids_val.add(ref["ann_id"])
    elif ref["split"] == "test":
        ids_test.add(ref["ann_id"])

for annotation in instances["annotations"]:
    if annotation["id"] in ids_train:
        new_dataset_train[annotation["id"]] = {}
        new_dataset_train[annotation["id"]]["image_id"] = annotation["image_id"]
        new_dataset_train[annotation["id"]]["bbox"] = annotation["bbox"]
    elif annotation["id"] in ids_test:
        new_dataset_test[annotation["id"]] = {}
        new_dataset_test[annotation["id"]]["image_id"] = annotation["image_id"]
        new_dataset_test[annotation["id"]]["bbox"] = annotation["bbox"]
    elif annotation["id"] in ids_val:
        new_dataset_val[annotation["id"]] = {}
        new_dataset_val[annotation["id"]]["image_id"] = annotation["image_id"]
        new_dataset_val[annotation["id"]]["bbox"] = annotation["bbox"]

for ref in refs:
    if ref["split"] == "train":
        sentences = ref["sentences"]
        new_dataset_train[ref["ann_id"]]["sentences"] = []
        for sentence in sentences:
            new_dataset_train[ref["ann_id"]]["sentences"].append(sentence["raw"])
        new_dataset_train[ref["ann_id"]]["category_id"] = ref["category_id"]
        image_name = (
            "COCO_train2014_"
            + str(new_dataset_train[ref["ann_id"]]["image_id"]).zfill(12)
            + ".jpg"
        )
        new_dataset_train[ref["ann_id"]]["image_name"] = image_name
    elif ref["split"] == "val":
        sentences = ref["sentences"]
        new_dataset_val[ref["ann_id"]]["sentences"] = []
        for sentence in sentences:
            new_dataset_val[ref["ann_id"]]["sentences"].append(sentence["raw"])
        new_dataset_val[ref["ann_id"]]["category_id"] = ref["category_id"]
        image_name = (
            "COCO_train2014_"
            + str(new_dataset_val[ref["ann_id"]]["image_id"]).zfill(12)
            + ".jpg"
        )
        new_dataset_val[ref["ann_id"]]["image_name"] = image_name
    elif ref["split"] == "test":
        sentences = ref["sentences"]
        new_dataset_test[ref["ann_id"]]["sentences"] = []
        for sentence in sentences:
            new_dataset_test[ref["ann_id"]]["sentences"].append(sentence["raw"])
        new_dataset_test[ref["ann_id"]]["category_id"] = ref["category_id"]
        image_name = (
            "COCO_train2014_"
            + str(new_dataset_test[ref["ann_id"]]["image_id"]).zfill(12)
            + ".jpg"
        )
        new_dataset_test[ref["ann_id"]]["image_name"] = image_name


new_json_object_train = json.dumps(new_dataset_train, indent=4)
new_json_object_test = json.dumps(new_dataset_test, indent=4)
new_json_object_val = json.dumps(new_dataset_val, indent=4)

with open(f"{get_config()['annotations_path']}/instances_train.json", "w") as outfile:
    outfile.write(new_json_object_train)

with open(f"{get_config()['annotations_path']}/instances_test.json", "w") as outfile:
    outfile.write(new_json_object_test)

with open(f"{get_config()['annotations_path']}/instances_val.json", "w") as outfile:
    outfile.write(new_json_object_val)
