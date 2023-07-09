from config import get_config
from models.DetachedHeadModel import (
    DetachedHeadModel,
    get_class_cost_function,
    get_class_optimizer,
    get_detect_cost_function,
    get_detect_optimizer,
    test_step,
    training_step,
)

# from models.FullHeadModel import (
#     FullHeadModel,
#     get_cost_function,
#     get_optimizer,
#     test_step,
#     training_step,
# )

from models.Metrics import Metrics
from clip import clip
from torch.utils.tensorboard import SummaryWriter
from models.CustomDataset import CustomDataset
import utils
import torch


def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


if __name__ == "__main__":
    device = get_config()["device"]

    learning_rate = 0.001
    weight_decay = 0.000001
    momentum = 0.9
    epochs = 30

    clip_model, preprocess = clip.load("RN50")
    clip_model = clip_model.eval()

    train_dataset = CustomDataset(split="train", model=clip_model, transform=preprocess)
    val_dataset = CustomDataset(split="val", model=clip_model, transform=preprocess)
    test_dataset = CustomDataset(split="test", model=clip_model, transform=preprocess)

    # metrics = Metrics(
    #     iou_threshold=0.5, prob_threshold=0.5, dataset_dim=len(val_dataset)
    # )

    best_detector_ever = DetachedHeadModel()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 64, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 64, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 64, shuffle=False, drop_last=True
    )

    writer = SummaryWriter(log_dir="runs/exp1")

    # instantiate the network and move it to the chosen device (GPU)
    net = DetachedHeadModel().to(get_config()["device"])
    print(net)

    # instantiate the optimizer
    detect_optimizer = get_detect_optimizer(net, learning_rate, weight_decay, momentum)
    class_optimizer = get_class_optimizer(net, learning_rate, weight_decay, momentum)

    # define the cost function
    detect_cost_function = get_detect_cost_function()
    class_cost_function = get_class_cost_function()

    # computes evaluation results before training
    print("Before training:")
    (
        detect_train_loss,
        detect_train_accuracy,
        class_train_loss,
        class_train_accuracy,
    ) = test_step(net, train_loader, detect_cost_function, class_cost_function)

    (
        detect_val_loss,
        detect_val_accuracy,
        class_val_loss,
        class_val_accuracy,
    ) = test_step(net, val_loader, detect_cost_function, class_cost_function)
    (
        detect_test_loss,
        detect_test_accuracy,
        class_test_loss,
        class_test_accuracy,
    ) = test_step(net, test_loader, detect_cost_function, class_cost_function)

    # log to TensorBoard
    log_values(
        writer, -1, detect_train_loss, detect_train_accuracy, "Detection Training"
    )
    log_values(writer, -1, detect_val_loss, detect_val_accuracy, "Detection Validation")
    log_values(writer, -1, detect_test_loss, detect_test_accuracy, "Detection Test")
    log_values(
        writer, -1, class_train_loss, class_train_accuracy, "Classification Training"
    )
    log_values(
        writer, -1, class_val_loss, class_val_accuracy, "Classification Validation"
    )
    log_values(writer, -1, class_test_loss, class_test_accuracy, "Classification Test")

    print(
        "\tDetection training loss {:.5f}, training accuracy {:.2f}".format(
            detect_train_loss, detect_train_accuracy
        )
    )

    print(
        "\tDetection validation loss {:.5f}, validation accuracy {:.2f}".format(
            detect_val_loss, detect_val_accuracy
        )
    )
    print(
        "\tDetection test loss {:.5f}, test accuracy {:.2f}".format(
            detect_test_loss, detect_test_accuracy
        )
    )

    print(
        "\tClassification training loss {:.5f}, training accuracy {:.2f}".format(
            class_train_loss, class_train_accuracy
        )
    )

    print(
        "\tClassification validation loss {:.5f}, validation accuracy {:.2f}".format(
            class_val_loss, class_val_accuracy
        )
    )
    print(
        "\tClassification test loss {:.5f}, test accuracy {:.2f}".format(
            class_test_loss, class_test_accuracy
        )
    )

    print("-----------------------------------------------------")

    # for each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        (
            detect_train_loss,
            detect_train_accuracy,
            class_train_loss,
            class_train_accuracy,
        ) = training_step(
            net,
            train_loader,
            detect_optimizer,
            detect_cost_function,
            class_optimizer,
            class_cost_function,
        )
        (
            detect_val_loss,
            detect_val_accuracy,
            class_val_loss,
            class_val_accuracy,
        ) = test_step(net, val_loader, detect_cost_function, class_cost_function)

        # logs to TensorBoard
        log_values(
            writer, e, detect_train_loss, detect_train_accuracy, "Detection Training"
        )
        log_values(
            writer, e, detect_val_loss, detect_val_accuracy, "Detection Validation"
        )
        log_values(
            writer, e, class_train_loss, class_train_accuracy, "Classification Training"
        )
        log_values(
            writer, e, class_val_loss, class_val_accuracy, "Classification Validation"
        )

        print("Epoch: {:d}".format(e + 1))
        print(
            "\tDetection training loss {:.5f}, training accuracy {:.2f}".format(
                detect_train_loss, detect_train_accuracy
            )
        )
        print(
            "\tDetection validation loss {:.5f}, validation accuracy {:.2f}".format(
                detect_val_loss, detect_val_accuracy
            )
        )
        print("-----------------------------------------------------")

    # compute final evaluation results
    print("After training:")
    (
        detect_train_loss,
        detect_train_accuracy,
        class_train_loss,
        class_train_accuracy,
    ) = test_step(net, train_loader, detect_cost_function, class_cost_function)

    (
        detect_val_loss,
        detect_val_accuracy,
        class_val_loss,
        class_val_accuracy,
    ) = test_step(net, val_loader, detect_cost_function, class_cost_function)
    (
        detect_test_loss,
        detect_test_accuracy,
        class_test_loss,
        class_test_accuracy,
    ) = test_step(net, test_loader, detect_cost_function, class_cost_function)

    # log to TensorBoard
    log_values(
        writer, epochs, detect_train_loss, detect_train_accuracy, "Detection Training"
    )
    log_values(
        writer, epochs, detect_val_loss, detect_val_accuracy, "Detection Validation"
    )
    log_values(writer, epochs, detect_test_loss, detect_test_accuracy, "Detection Test")
    log_values(
        writer,
        epochs,
        class_train_loss,
        class_train_accuracy,
        "Classification Training",
    )
    log_values(
        writer, epochs, class_val_loss, class_val_accuracy, "Classification Validation"
    )
    log_values(
        writer, epochs, class_test_loss, class_test_accuracy, "Classification Test"
    )

    print(
        "\tDetection training loss {:.5f}, training accuracy {:.2f}".format(
            detect_train_loss, detect_train_accuracy
        )
    )

    print(
        "\tDetection validation loss {:.5f}, validation accuracy {:.2f}".format(
            detect_val_loss, detect_val_accuracy
        )
    )
    print(
        "\tDetection test loss {:.5f}, test accuracy {:.2f}".format(
            detect_test_loss, detect_test_accuracy
        )
    )

    print(
        "\tClassification training loss {:.5f}, training accuracy {:.2f}".format(
            class_train_loss, class_train_accuracy
        )
    )

    print(
        "\tClassification validation loss {:.5f}, validation accuracy {:.2f}".format(
            class_val_loss, class_val_accuracy
        )
    )
    print(
        "\tClassification test loss {:.5f}, test accuracy {:.2f}".format(
            class_test_loss, class_test_accuracy
        )
    )

    # closes the logger
    writer.close()

    torch.save(net.state_dict(), "best-ever.pt")
