import torch
import torch.nn as nn
import torchvision
import clip
import math

from config import get_config


class FullHeadModel(nn.Module):
    def __init__(self):
        super().__init__()

        clip_model, _ = clip.load("RN50")
        self.clip_vision_model = clip_model.visual

        for param in self.clip_vision_model.parameters():
            param.requires_grad = False

        self.clip_text_model = clip_model.encode_text

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
    cost_function = torch.nn.MSELoss()
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

