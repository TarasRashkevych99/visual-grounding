import pickle
import matplotlib.pyplot as plt
from PIL import Image

annotations_path = "./refcocog/annotations/refs(umd).p"

annotations = pickle.load(open(annotations_path, "rb"))

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

plt.subplot(1, 2, 1)
im1 = Image.open("./refcocog/images/COCO_train2014_000000041700_197196.jpg")
im2 = Image.open("./refcocog/images/COCO_train2014_000000041700_191423.jpg")
plt.imshow(im1)
plt.subplot(1, 2, 2)
plt.imshow(im2)
plt.show()

print(len(ids))