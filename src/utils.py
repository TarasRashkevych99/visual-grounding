import matplotlib.pyplot as plt
import matplotlib.patches as patches
from clip import clip
import torch
import numpy as np

from config import get_config


def plot_bounding_boxes(image, boxes, indeces, ground_bbox=None):
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
    if ground_bbox:
        rect = plt.Rectangle(
            (ground_bbox[0], ground_bbox[1]),
            ground_bbox[2],
            ground_bbox[3],
            linewidth=1,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)
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


def encode_data_with_clip(clip_model, image, text):
    text_tokens = clip.tokenize(text).to(get_config()["device"])
    with torch.no_grad():
        image_z = (
            clip_model.encode_image(image.unsqueeze(0))
            .float()
            .to(get_config()["device"])
        )
        text_z = clip_model.encode_text(text_tokens).float().to(get_config()["device"])

    image_z /= image_z.norm(dim=-1, keepdim=True)
    text_z /= text_z.norm(dim=-1, keepdim=True)
    # image_z_stack = image_z.expand(50, -1)
    # text_z_stack = text_z.expand(50, -1)
    # plt.subplot(2, 1, 1)
    # plt.imshow(image_z_stack.cpu().numpy(), cmap="gray")
    # plt.subplot(2, 1, 2)
    # plt.imshow(text_z_stack.cpu().numpy(), cmap="gray")
    # plt.show()
    # plt.figure()
    # plt.imshow(torch.mm(image_z.T, text_z).cpu().numpy(), cmap="gray")
    # plt.show()
    return torch.mm(image_z.T, text_z)


def crop_image_by_boxes(image, boxes):
    cropped_images = []
    for index, xyxy in enumerate(boxes.xyxy.to("cpu")):
        cropped_img = image.crop(xyxy.numpy())
        cropped_images.append(cropped_img)
    return cropped_images


def preprocess_images(images, preprocess):
    preprocessed_images = []
    for image in images:
        preprocessed_images.append(preprocess(image).to(get_config()["device"]))
    return preprocessed_images


def get_probs(images_z, texts_z):
    images_z /= images_z.norm(dim=-1, keepdim=True)
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    probs = (100 * images_z @ texts_z.T).softmax(dim=0)
    return probs


def get_threshold(probs):
    threshold = sum(probs[:]) / len(probs[:])
    return threshold


def xywh_to_xyxy(xywh):
    xyxy = np.zeros(4)
    xyxy[0] = xywh[0]
    xyxy[1] = xywh[1]
    xyxy[2] = xywh[0] + xywh[2]
    xyxy[3] = xywh[1] + xywh[3]
    return xyxy
