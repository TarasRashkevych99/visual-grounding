import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.patches as patches

total_count = 0
boxes_count = 0

bbox = json.load(open("refcocog/annotations/instances_bbox.json"))
path_to_images = 'refcocog/images'
for filename in os.listdir(path_to_images):
    if filename.endswith('.jpg'):
        filename_truncated = filename[-13:]
        first_non_zero_index = next((i for i, c in enumerate(filename_truncated) if c != '0'), None)
        image_id = filename_truncated[first_non_zero_index:-4]
        if image_id in bbox.keys():
            boxes_count += 1
            _, ax = plt.subplots()
            path_to_image = f"{path_to_images}/{filename}"
            im = Image.open(path_to_image)
            ax.imshow(im)
            for box in bbox[image_id]:
                rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()
        else:
            print(f"Image {filename} not found in bbox")
        total_count += 1

print("Total images: " + str(total_count))
print("Images with boxes: " + str(boxes_count))

