import pickle
import matplotlib.pyplot as plt
from PIL import Image
import json

annotations_path = "./refcocog/annotations/refs(umd).p"

annotations = pickle.load(open(annotations_path, "rb"))
instances = json.load(open("refcocog/annotations/instances.json"))

ids = set()
for annotaition in annotations:
    if annotaition["image_id"] in ids:
        print(ids)
        print("Duplicated: ", annotaition["image_id"])
        break
    ids.add(annotaition["image_id"])

for annotaition in annotations:
    if annotaition["image_id"] == 41700:
        print(annotaition)



for i in instances['annotations']:
    if i['id'] == 197196:
        bbox1 = i['bbox']
    if i['id'] == 191423:
        bbox2 = i['bbox']

_, ax = plt.subplots()
im1 = Image.open("./refcocog/images/COCO_train2014_000000041700.jpg")
plt.imshow(im1)
rect1 = plt.Rectangle((bbox1[0], bbox1[1]), bbox1[2], bbox1[3], linewidth=1, edgecolor='g', facecolor='none')
rect2 = plt.Rectangle((bbox2[0], bbox2[1]), bbox2[2], bbox2[3], linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)
plt.show()

print(len(ids))