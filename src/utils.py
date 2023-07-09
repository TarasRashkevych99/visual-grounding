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

    # image_z /= image_z.norm(dim=-1, keepdim=True)
    # text_z /= text_z.norm(dim=-1, keepdim=True)
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


def xywh_to_topleft(xywh):
    x, y, w, h = xywh.tolist()
    x1 = x - w / 2
    y1 = y - h / 2
    return torch.tensor([x1, y1, w, h])


categories = {
    1: "Person",
    2: "Bicycle",
    3: "Car",
    4: "Motorcycle",
    5: "Airplane",
    6: "Bus",
    7: "Train",
    8: "Truck",
    9: "Boat",
    10: "Traffic light",
    11: "Fire hydrant",
    12: "Street sign",
    13: "Stop sign",
    14: "Parking meter",
    15: "Bench",
    16: "Bird",
    17: "Cat",
    18: "Dog",
    19: "Horse",
    20: "Sheep",
    21: "Cow",
    22: "Elephant",
    23: "Bear",
    24: "Zebra",
    25: "Giraffe",
    26: "Hat",
    27: "Backpack",
    28: "Umbrella",
    29: "Shoe",
    30: "Eye glasses",
    31: "Handbag",
    32: "Tie",
    33: "Suitcase",
    34: "Frisbee",
    35: "Skis",
    36: "Snowboard",
    37: "Sports ball",
    38: "Kite",
    39: "Baseball bat",
    40: "Baseball glove",
    41: "Skateboard",
    42: "Surfboard",
    43: "Tennis racket",
    44: "Bottle",
    45: "Plate",
    46: "Wine glass",
    47: "Cup",
    48: "Fork",
    49: "Knife",
    50: "Spoon",
    51: "Bowl",
    52: "Banana",
    53: "Apple",
    54: "Sandwich",
    55: "Orange",
    56: "Broccoli",
    57: "Carrot",
    58: "Hot dog",
    59: "Pizza",
    60: "Donut",
    61: "Cake",
    62: "Chair",
    63: "Couch",
    64: "Potted plant",
    65: "Bed",
    66: "Mirror",
    67: "Dining table",
    68: "Window",
    69: "Desk",
    70: "Toilet",
    71: "Door",
    72: "TV",
    73: "Laptop",
    74: "Mouse",
    75: "Remote",
    76: "Keyboard",
    77: "Cell phone",
    78: "Microwave",
    79: "Oven",
    80: "Toaster",
    81: "Sink",
    82: "Refrigerator",
    83: "Blender",
    84: "Book",
    85: "Clock",
    86: "Vase",
    87: "Scissors",
    88: "Teddy bear",
    89: "Hair drier",
    90: "Toothbrush",
    91: "Hair brush",
}
