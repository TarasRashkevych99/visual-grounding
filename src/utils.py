import matplotlib.pyplot as plt
import matplotlib.patches as patches
from clip import clip
import torch
import numpy as np
from models.DatasetSplit import DatasetSplit
from models.AnnotationSplitter import AnnotationSplitter
import json
from PIL import Image


def plot_bounding_boxes(image, boxes, indeces, image_name=None):
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
            if image_name:
                plot_ground_bbox(ax, image_name)
            # l.set_bbox(dict(facecolor="red", alpha=0.5, edgecolor="red"))
    #plt.show()


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


def get_partitions(transform=None):
    annotation_splitter = AnnotationSplitter()
    train = DatasetSplit(annotation_splitter.train_set_annotations.copy(), transform)
    val = DatasetSplit(annotation_splitter.val_set_annotations.copy(), transform)
    test = DatasetSplit(annotation_splitter.test_set_annotations.copy(), transform)
    return train, val, test

def plot_ground_bbox(image_name, bbox):
    path_to_images = 'refcocog/images'
    #bbox = json.load(open("refcocog/annotations/instances_bbox.json"))
    filename_truncated = image_name[-13:]
    first_non_zero_index = next((i for i, c in enumerate(filename_truncated) if c != '0'), None)
    image_id = filename_truncated[first_non_zero_index:-4]
    if image_id in bbox.keys():
        # im = Image.open(image_name)
        # ax.imshow(im)
        # for box in bbox[image_id]:
        #     rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='g', facecolor='none')
        #     ax.add_patch(rect)
        print(f"Image {image_id} found in bbox")
    else:
        print(f"Image {image_name} not found in bbox")
