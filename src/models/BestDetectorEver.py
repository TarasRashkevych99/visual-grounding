import torch
import torch.nn as nn
import clip
import math

from config import get_config


class BestDetectorEver(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_vision_model = get_clip_model()

        for param in self.clip_vision_model.parameters():
            param.requires_grad = False

        clip_model, _ = clip.load("RN50")
        self.clip_text_model = clip_model.encode_text

        self.reduce_dimensionality = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Conv2d(2048, 1024, kernel_size=4, stride=4),
            nn.Flatten(),
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
        images = self.clip_vision_model(images)
        texts = self.clip_text_model(texts)
        images = self.reduce_dimensionality(images)
        embeddings = torch.cat((images, texts), dim=1)
        bbox = self.bbox_regression(embeddings)
        return bbox
    
    def _create_positional_encoding(self):
        # Create positional encodings for sequence length
        pe = torch.zeros(self.sequence_length, self.input_dim)
        position = torch.arange(0, self.sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_dim, 2).float() * (-math.log(10000.0) / self.input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


def get_optimizer(model, lr, wd, momentum):
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr / 10,
    #     weight_decay=wd,
    #     momentum=momentum,
    # )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )
    return optimizer


def get_cost_function():
    #cost_function = torch.nn.MSELoss()
    cost_function = torch.nn.L1Loss()
    return cost_function


def training_step(net, data_loader, optimizer, cost_function):
    samples = 0.0
    cumulative_loss = 0.0
    localization_accuracy = 0.0
    counter = 0
    # set the network to training mode
    net.train()

    # iterate over the training set
    for images, texts, bboxes, category_id in data_loader:
        counter += 1
        print("Batch: ", counter)
        # load data into GPU
        #embeddings = embeddings.to(get_config()["device"]).unsqueeze(1)
        bboxes = bboxes.to(get_config()["device"])
        # forward pass
        outputs = net(images, texts).to(get_config()["device"])

        # loss computation
        loss = cost_function(outputs, bboxes)

        # backward pass
        loss.backward()

        print("Loss: ", loss)

        # parameters update
        optimizer.step()

        # gradients reset
        optimizer.zero_grad()

        # fetch prediction and loss value
        samples += images.shape[0]
        cumulative_loss += loss.item()
        predicted = outputs

        # compute training accuracy
        localization_accuracy += sum([compute_iou(pred, bboxes[index]) for index, pred in enumerate(predicted)])

    return cumulative_loss / samples, localization_accuracy / samples * 100


def test_step(net, data_loader, cost_function):
    samples = 0.0
    cumulative_loss = 0.0
    localization_accuracy = 0.0

    # set the network to evaluation mode
    net.eval()
    counter = 0
    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for images, texts, bboxes, category_id in data_loader:
            counter += 1
            print("Batch: ", counter)
            # load data into GPU
            #embeddings = embeddings.to(get_config()["device"]).unsqueeze(1)
            bboxes = bboxes.to(get_config()["device"])
            # forward pass
            outputs = net(images, texts).to(get_config()["device"])

            # loss computation
            loss = cost_function(outputs, bboxes)

            # fetch prediction and loss value
            samples += images.shape[0]
            cumulative_loss += loss.item()
            # Note: the .item() is needed to extract scalars from tensors
            predicted = outputs

            # compute accuracy
            localization_accuracy += sum([compute_iou(pred, bboxes[index]) for index, pred in enumerate(predicted)])

    return cumulative_loss / samples, localization_accuracy / samples * 100


def compute_iou(predicted_box, ground_box):
    # print(predicted_box)
    # print(ground_box)
    x1 = max(predicted_box[0], ground_box[0])
    y1 = max(predicted_box[1], ground_box[1])
    x2 = min(predicted_box[2], ground_box[2])
    y2 = min(predicted_box[3], ground_box[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (predicted_box[2] - predicted_box[0] + 1) * (
        predicted_box[3] - predicted_box[1] + 1
    )
    box2_area = (ground_box[2] - ground_box[0] + 1) * (
        ground_box[3] - ground_box[1] + 1
    )

    # Calculate the IoU

    if intersection_area > 0:
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    else:
        return 0.0

def get_clip_model():
    clip_model, _ = clip.load("RN50")
    clip_vision_model = clip_model.visual
    layers = list(clip_vision_model.children())
    vision_model = nn.Sequential(*layers[:-1])
    return vision_model.float().to(get_config()["device"])

