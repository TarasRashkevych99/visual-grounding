from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from clip import clip
import torch
import os
import numpy as np
import cv2

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


def plot_bounding_boxes(img_path, results, indeces):
    img = Image.open(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for result in results:
        for index, xywh in enumerate(result.boxes.xywh):
            print(result.boxes[index].cls)
            if index in indeces:
                rect = patches.Rectangle(
                    (xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2),
                    xywh[2],
                    xywh[3],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                    label=f"{result.boxes[index].cls}",
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
    model = YOLO("yolov8n.pt")

    clip_model, preprocess = clip.load("RN50")
    clip_model = clip_model.eval()

    texts = [
        # "People walking on the street",
        # "Two people walking",
        # "Road sign",
        "Person with black coat",
    ]
    images = []

    results = model("bus.jpg")

    im = Image.open("bus.jpg")
    for result in results:
        for index, xyxy in enumerate(result.boxes.xyxy):
            cropped_img = im.crop(xyxy.numpy())
            images.append(preprocess(cropped_img))

    # for filename in os.listdir("crops"):
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         file_path = os.path.join("crops", filename)
    #         print(file_path)
    #         image = Image.open(file_path)
    #         image = preprocess(image)
    #         images.append(image)

    images_z, texts_z = clip_encoder(images, texts)

    images_z /= images_z.norm(dim=-1, keepdim=True)
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    outputs = (100 * images_z @ texts_z.T).softmax(dim=0)

    threshold = sum(outputs[:]) / len(outputs[:])
    print(threshold)
    print(outputs)

    obj_indeces = [index for (index, prob) in enumerate(outputs[:]) if prob > threshold]

    # res_plotted = results[0].plot()
    # cv2.imshow("result", res_plotted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plot_bounding_boxes("bus.jpg", results, obj_indeces)

    _, predicted = outputs.max(1)

    for i in images_z:
        for j in texts_z:
            print(f"{float(cosine_similarity(i,j)):.3f}", end="|")
        print()
