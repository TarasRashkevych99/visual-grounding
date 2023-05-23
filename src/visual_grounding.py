from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from clip import clip
import torch
import numpy as np
from utils import get_partitions


def plot_bounding_boxes(image, boxes, indeces):
    _, ax = plt.subplots()
    ax.imshow(image)
    for index, xywh in enumerate(boxes.xywh):
        if index in indeces:
            anchor_point_x, anchor_point_y = (
                xywh[0] - xywh[2] / 2,
                xywh[1] - xywh[3] / 2,
            )
            width, height = (xywh[2], xywh[3])
            rect = patches.Rectangle(
                (anchor_point_x, anchor_point_y),
                width,
                height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            l = ax.annotate(
                f"{int(boxes[index].cls.item())}",
                (anchor_point_x, anchor_point_y),
                fontsize=8,
                fontweight="bold",
                color="white",
                ha="left",
                va="top",
            )
            # l.set_bbox(dict(facecolor="red", alpha=0.5, edgecolor="red"))
    plt.show()


def print_cosine_similarity_matrix(images_z, texts_z):
    for i in images_z:
        for j in texts_z:
            print(f"{float(cosine_similarity(i,j)):.3f}", end="|")
        print()


def cosine_similarity(images_z: torch.Tensor, texts_z: torch.Tensor):
    # normalise the image and the text
    images_z /= images_z.norm(dim=-1, keepdim=True)
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    # evaluate the cosine similarity between the sets of features
    similarity = texts_z @ images_z.T

    return similarity.cpu()


def encode_data_with_clip(clip_model, images, texts):
    images = torch.tensor(np.stack(images))
    text_tokens = clip.tokenize(texts)
    with torch.no_grad():
        images_z = clip_model.encode_image(images).float()
        texts_z = clip_model.encode_text(text_tokens).float()

    return images_z, texts_z


def crop_image_by_boxes(image, boxes):
    cropped_images = []
    for index, xyxy in enumerate(boxes.xyxy):
        cropped_img = image.crop(xyxy.numpy())
        cropped_images.append(cropped_img)
    return cropped_images


def preprocess_images(images, preprocess):
    preprocessed_images = []
    for image in images:
        preprocessed_images.append(preprocess(image))
    return preprocessed_images


def get_probs(images_z, texts_z):
    images_z /= images_z.norm(dim=-1, keepdim=True)
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    probs = (100 * images_z @ texts_z.T).softmax(dim=0)
    return probs


def get_threshold(probs):
    threshold = sum(probs[:]) / len(probs[:])
    return threshold


if __name__ == "__main__":
    yolo_model = YOLO("yolov8n.pt")

    clip_model, preprocess = clip.load("RN50")
    clip_model = clip_model.eval()

    # texts = [
    #     # "People walking on the street",
    #     # "Two people walking",
    #     # "Road sign",
    #     # "Person with a black coat",
    #     "Person with a yellow coat",
    #     # "Person with a yellow coat and a person with a black coat",
    # ]

    train, val, test = get_partitions()

    for image, random_sentence in val:
        results = yolo_model(image)
        boxes = results[0].boxes

        cropped_images = crop_image_by_boxes(image, boxes)
        preprocessed_images = preprocess_images(cropped_images, preprocess)
        images_z, texts_z = encode_data_with_clip(
            clip_model, preprocessed_images, random_sentence
        )

        # print_cosine_similarity_matrix(images_z, texts_z)

        probs = get_probs(images_z, texts_z)

        threshold = get_threshold(probs)

        print(random_sentence)
        print(threshold)
        print(probs)

        obj_indeces = [
            index for (index, prob) in enumerate(probs[:]) if prob > threshold
        ]

        plot_bounding_boxes(image, boxes, obj_indeces)
