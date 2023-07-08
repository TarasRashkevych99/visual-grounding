from config import get_config
from models.BestDetectorEver import (
    BestDetectorEver,
    get_cost_function,
    get_optimizer,
    test_step,
    training_step,
)
from models.Metrics import Metrics
from clip import clip
# from torch.utils.tensorboard import SummaryWriter
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
    epochs = 50

    clip_model, preprocess = clip.load("RN50")
    clip_model = clip_model.eval()

    train_dataset = CustomDataset(
        split="train", model=clip_model, transform=preprocess
    )
    val_dataset = CustomDataset(
        split="val", model=clip_model, transform=preprocess
    )
    test_dataset = CustomDataset(split="test", model=clip_model, transform=preprocess)

    # metrics = Metrics(
    #     iou_threshold=0.5, prob_threshold=0.5, dataset_dim=len(val_dataset)
    # )

    best_detector_ever = BestDetectorEver()

    train_loader = torch.utils.data.DataLoader(train_dataset, 64, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 64, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 64, shuffle=False, drop_last=True)

    # writer = SummaryWriter(log_dir="runs/exp1")

    # instantiate the network and move it to the chosen device (GPU)
    net = BestDetectorEver().to(get_config()["device"])

    # instantiate the optimizer
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    # define the cost function
    cost_function = get_cost_function()

    # computes evaluation results before training
    print("Before training:")
    #train_loss, train_accuracy = test_step(net, train_loader, cost_function)
    val_loss, val_accuracy = test_step(net, val_loader, cost_function)
    test_loss, test_accuracy = test_step(net, test_loader, cost_function)

    # log to TensorBoard
    # log_values(writer, -1, train_loss, train_accuracy, "train")
    # log_values(writer, -1, val_loss, val_accuracy, "validation")
    # log_values(writer, -1, test_loss, test_accuracy, "test")

    
    print(
        "\tValidation loss {:.5f}, Validation accuracy {:.2f}".format(
            val_loss, val_accuracy
        )
    )
    print("\tTest loss {:.5f}, Test accuracy {:.2f}".format(test_loss, test_accuracy))
    print("-----------------------------------------------------")

    # for each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        train_loss, train_accuracy = training_step(
            net, train_loader, optimizer, cost_function
        )
        val_loss, val_accuracy = test_step(net, val_loader, cost_function)

        # logs to TensorBoard
        # log_values(writer, e, val_loss, val_accuracy, "Validation")

        print("Epoch: {:d}".format(e + 1))
        print(
            "\tTraining loss during training{:.5f}, Training accuracy {:.2f}".format(
                train_loss, train_accuracy
            )
        )
        print(
            "\tValidation loss during training{:.5f}, Validation accuracy {:.2f}".format(
                val_loss, val_accuracy
            )
        )
        print("-----------------------------------------------------")

    # compute final evaluation results
    print("After training:")
    train_loss, train_accuracy = test_step(net, train_loader, cost_function)
    val_loss, val_accuracy = test_step(net, val_loader, cost_function)
    test_loss, test_accuracy = test_step(net, test_loader, cost_function)

    # log to TensorBoard
    # log_values(writer, epochs, train_loss, train_accuracy, "train")
    # log_values(writer, epochs, val_loss, val_accuracy, "validation")
    # log_values(writer, epochs, test_loss, test_accuracy, "test")

    print(
        "\tTraining loss {:.5f}, Training accuracy {:.2f}".format(
            train_loss, train_accuracy
        )
    )
    print(
        "\tValidation loss {:.5f}, Validation accuracy {:.2f}".format(
            val_loss, val_accuracy
        )
    )
    print("\tTest loss {:.5f}, Test accuracy {:.2f}".format(test_loss, test_accuracy))
    print("-----------------------------------------------------")

    # closes the logger
    # writer.close()

    # Quantize the model
    # torch.quantization.prepare(net, inplace=True)
    # torch.quantization.convert(net, inplace=True)

    torch.save(net.state_dict(), 'best-ever.pt')