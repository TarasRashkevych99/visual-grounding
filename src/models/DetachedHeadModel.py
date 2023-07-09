import torch
import torch.nn as nn
import torchvision
import clip
import math

from config import get_config


class DetachedHeadModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_vision_model = get_clip_model()

        for param in self.clip_vision_model.parameters():
            param.requires_grad = False

        clip_model, _ = clip.load("RN50")
        self.clip_text_model = clip_model.encode_text

        clip_for_class, _ = clip.load("RN50")
        self.classifier_text_backbone = clip_for_class.encode_text
        self.classifier_visual_backbone = clip_for_class.visual

        for param in self.classifier_visual_backbone.parameters():
            param.requires_grad = False

        self.classifier_head = torch.nn.Linear(2048, 91)

        self.reduce_dimensionality = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(18432, 1024),
        )

        self.bbox_regression = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, images, texts):
        encoded_texts = self.classifier_text_backbone(texts).float()
        encoded_images = self.classifier_visual_backbone(images)
        unbounded_class_probs = self.classifier_head(
            torch.cat((encoded_images, encoded_texts), dim=1)
        )
        images = self.clip_vision_model(images)
        texts = self.clip_text_model(texts)
        images = self.reduce_dimensionality(images)
        embeddings = torch.cat((images, texts), dim=1)
        bbox = self.bbox_regression(embeddings)
        return bbox.to(get_config()["device"]), unbounded_class_probs.to(
            get_config()["device"]
        )

    def __str__(self) -> str:
        return "DetachedHeadModel"


def get_detect_optimizer(model, lr, wd):
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr / 10,
    #     weight_decay=wd,
    #     momentum=momentum,
    # )
    params = list(model.reduce_dimensionality.parameters()) + list(
        model.bbox_regression.parameters()
    )
    optimizer = torch.optim.Adam(
        params,
        lr=lr,
        weight_decay=wd,
    )
    return optimizer


def get_class_optimizer(model, lr, wd, momentum):
    optimizer = torch.optim.SGD(
        [{"params": model.classifier_head.parameters(), "lr": lr}],
        lr=lr / 10,
        weight_decay=wd,
        momentum=momentum,
    )

    return optimizer


def get_detect_cost_function():
    cost_function = torch.nn.MSELoss()
    return cost_function


def get_class_cost_function():
    cost_function = torch.nn.CrossEntropyLoss()
    return cost_function


def training_step(
    net,
    data_loader,
    detect_optimizer,
    detect_cost_function,
    class_optimizer,
    class_cost_function,
):
    samples = 0.0
    cumulative_detect_loss = 0.0
    cumulative_class_loss = 0.0
    localization_accuracy = 0.0
    class_accuracy = 0.0
    counter = 0
    # set the network to training mode
    net.train()

    # iterate over the training set
    for images, texts, bboxes, category_ids in data_loader:
        counter += 1
        print("Batch: ", counter)
        # load data into GPU
        bboxes = bboxes.to(get_config()["device"])
        # gradients reset
        detect_optimizer.zero_grad()
        class_optimizer.zero_grad()
        # forward pass
        predicted_bboxes, unbound_class_probs = net(images, texts)

        # loss computation
        detect_loss = detect_cost_function(predicted_bboxes, bboxes)
        class_loss = class_cost_function(unbound_class_probs, category_ids)

        # backward pass
        detect_loss.backward()
        class_loss.backward()

        print("Detect Loss: ", detect_loss)
        print("Class Loss: ", class_loss)

        # parameters update
        detect_optimizer.step()
        class_optimizer.step()

        # fetch prediction and loss value
        samples += images.shape[0]
        cumulative_detect_loss += detect_loss.item()
        cumulative_class_loss += class_loss.item()

        # compute training accuracy
        localization_accuracy += sum(
            [
                compute_iou(pred, bboxes[index])
                for index, pred in enumerate(predicted_bboxes)
            ]
        )

        _, predicted_classes = unbound_class_probs.max(
            dim=1
        )  # max() returns (maximum_value, index_of_maximum_value)

        # compute training accuracy
        class_accuracy += predicted_classes.eq(category_ids).sum().item()

    return (
        cumulative_detect_loss / samples,
        localization_accuracy / samples * 100,
        cumulative_class_loss / samples,
        class_accuracy / samples * 100,
    )


def test_step(
    net,
    data_loader,
    detect_cost_function,
    class_cost_function,
):
    samples = 0.0
    cumulative_detect_loss = 0.0
    cumulative_class_loss = 0.0
    localization_accuracy = 0.0
    class_accuracy = 0.0

    # set the network to evaluation mode
    net.eval()
    counter = 0
    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for images, texts, bboxes, category_ids in data_loader:
            # print("Category Ids: ", category_ids.shape)
            # exit()
            counter += 1
            print("Batch: ", counter)
            # load data into GPU
            bboxes = bboxes.to(get_config()["device"])

            # forward pass
            predicted_bboxes, unbound_class_probs = net(images, texts)

            # loss computation
            detect_loss = detect_cost_function(predicted_bboxes, bboxes)
            class_loss = class_cost_function(unbound_class_probs, category_ids)

            # fetch prediction and loss value
            samples += images.shape[0]
            cumulative_detect_loss += detect_loss.item()
            cumulative_class_loss += class_loss.item()

            # compute training accuracy
            localization_accuracy += sum(
                [
                    compute_iou(pred, bboxes[index])
                    for index, pred in enumerate(predicted_bboxes)
                ]
            )

            _, predicted_classes = unbound_class_probs.max(
                dim=1
            )  # max() returns (maximum_value, index_of_maximum_value)

            # compute training accuracy
            class_accuracy += predicted_classes.eq(category_ids).sum().item()

    return (
        cumulative_detect_loss / samples,
        localization_accuracy / samples * 100,
        cumulative_class_loss / samples,
        class_accuracy / samples * 100,
    )


def compute_iou(predicted_box, ground_box):
    predicted_box = predicted_box.squeeze(0)
    x, y, w, h = predicted_box.tolist()
    x1_min = x - w / 2
    y1_min = y - h / 2
    x1_max = x + w / 2
    y1_max = y + h / 2

    x, y, w, h = ground_box.tolist()
    x2_min = x - w / 2
    y2_min = y - h / 2
    x2_max = x + w / 2
    y2_max = y + h / 2

    # Calculate intersection coordinates
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Calculate intersection area
    intersection_width = max(0, x_inter_max - x_inter_min)
    intersection_height = max(0, y_inter_max - y_inter_min)
    intersection_area = intersection_width * intersection_height

    # Calculate box areas
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def torch_iou(predicted_box, ground_box):
    predicted_box = predicted_box.squeeze(0)
    x, y, w, h = predicted_box.tolist()
    x1_min = x - w / 2
    y1_min = y - h / 2
    x1_max = x + w / 2
    y1_max = y + h / 2

    box_1 = torch.tensor([x1_min, y1_min, x1_max, y1_max])

    x, y, w, h = ground_box.tolist()
    x2_min = x - w / 2
    y2_min = y - h / 2
    x2_max = x + w / 2
    y2_max = y + h / 2

    box_2 = torch.tensor([x2_min, y2_min, x2_max, y2_max])

    return torchvision.ops.box_iou(box_1.unsqueeze(0), box_2.unsqueeze(0))


def get_clip_model():
    clip_model, _ = clip.load("RN50")
    clip_vision_model = clip_model.visual
    layers = list(clip_vision_model.children())
    vision_model = nn.Sequential(*layers[:-1])
    return vision_model.float().to(get_config()["device"])
