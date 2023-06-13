import json
import matplotlib.pyplot as plt
from PIL import Image
import os

instances = json.load(open("refcocog/annotations/instances.json"))

for key in instances.keys():
    print(key)
    if isinstance(instances[key], dict):
        for k in instances[key].keys():
            print("  " + k)
            print("  " + str(type(instances[key][k])))
    print(type(instances[key]))
    print(len(instances[key]))

print("Example:")
print(instances["info"])
for key in instances.keys():
    if (key != "info"):
        print(instances[key][5])

count_images = 0 
ids = set()
for i in instances['annotations']:
    #if str(i['image_id']) not in ids:
    ids.add(str(i['image_id']))
    count_images += 1

print("Images number: " + str(count_images))
print("Ids lenght: " + str(len(ids)))

for cat in instances['categories']:
    print(cat['name'] + ', ', end='')

# name_suffix = str(instances['annotations'][5]['image_id']) + '.jpg'

# for root, dirs, files in os.walk("refcocog/images"):
#     for file in files:
#         if file.endswith(name_suffix):
#             path_to_image = os.path.join(root, file)


# new_json = {}

# for i in ids:
#     new_json[i] = []

# print(new_json)

# for i in instances['annotations']:
#     new_json[i['image_id']].append(i['bbox'])

# print(new_json)

# new_json_object = json.dumps(new_json, indent = 4)

# with open("refcocog/annotations/instances_bbox.json", "w") as outfile:
#     outfile.write(new_json_object)

# path_to_image = "refcocog/images/" + instances['images'][3]['file_name']
# print(path_to_image)
# im = Image.open(path_to_image)

# _, ax = plt.subplots()
# ax.imshow(im)
# plt.show()