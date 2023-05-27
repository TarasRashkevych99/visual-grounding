import json 
from PIL import Image
import matplotlib.pyplot as plt

instances_train = json.load(open("refcocog/annotations/instances_test.json"))

for key in instances_train.keys():
    print(key)
    print(instances_train[key])
    _, ax = plt.subplots()
    im = Image.open("./refcocog/images/" + instances_train[key]["image_name"])
    plt.imshow(im)
    bbox = instances_train[key]["bbox"]
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    plt.show()