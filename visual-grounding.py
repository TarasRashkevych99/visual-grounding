from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from clip import clip
import torch
import os
import numpy as np

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
clip_model, preprocess = clip.load("RN50")

clip_model = clip_model.eval()
# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# success = model.export(format="onnx")  # export the model to ONNX format


# im = Image.open("bus.jpg")
# for result in results:
#     # print(f"boxes: {result.boxes}")
#     for index, xyxy in enumerate(result.boxes.xyxy):

#         numpy_xyxy = xyxy.numpy()

#         cropped_img = im.crop(
#             (numpy_xyxy[0], numpy_xyxy[1], numpy_xyxy[2], numpy_xyxy[3])
#         )
#         cropped_img.save(f"cropped_img_{index}.jpg")
#         # print(f"xywh: {xywh}")
#         # print(f"masks: {result.masks}")
#         # print(f"probs: {result.probs}")


def plot_bounding_boxes(results):
    fig, ax = plt.subplots()
    ax.imshow(im)
    for result in results:
        for xywh in result.boxes.xywh:
            rect = patches.Rectangle(
                (xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2),
                xywh[2],
                xywh[3],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
    plt.show()


def clip_encoder(images, texts):
    images = torch.tensor(np.stack(images))
    text_tokens = clip.tokenize(texts)

    with torch.no_grad():
        images_z = clip_model.encode_image(images).float()
        texts_z = clip_model.encode_text(text_tokens).float()

    return images_z, texts_z


def cosine_similarity(images_z: torch.Tensor, texts_z: torch.Tensor):
    # normalise the image and the text
    images_z /= images_z.norm(dim=-1, keepdim=True)
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    # evaluate the cosine similarity between the sets of features
    similarity = texts_z @ images_z.T

    return similarity.cpu()


if __name__ == "__main__":

    texts = ["Two people are standing in front of a bus.", "Road sign"]
    images = []
    for filename in os.listdir("crops"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join("crops", filename)
            image = Image.open(file_path)
            image = preprocess(image)
            images.append(image)

    images_z, texts_z = clip_encoder(images, texts)

    for i in images_z:
        for j in texts_z:
            print(f"{float(cosine_similarity(i,j)):.3f}", end="|")
        print()
