import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as torchvision
from custom_data import Winding_Dataset
import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def get_data():
    # Load FashionMNIST dataset.
    dir_path = "/scratch/mrvl005h/Image_data"
    train_dataset = Winding_Dataset(
        csv_file="data-csv/unique_train.csv", root_dir=dir_path
    )
    valid_dataset = Winding_Dataset(
        csv_file="data-csv/unique_val.csv", root_dir=dir_path
    )
    test_dataset = Winding_Dataset(
        csv_file="data-csv/linear_winding_labels_image_level_test_set_v2.0.csv", root_dir=dir_path
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCHSIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCHSIZE, shuffle=True
    )
    print("data loaded")
    return train_loader, test_loader,valid_loader



def test_model(model, test_loader, loss_fn, epoch, writer):
    print(f"testing epoch: {epoch}")

    model.eval()
    all_preds = []
    all_labels = []
    test_loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_loader)):
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            model_output = model(data)
            model_output = F.sigmoid(model_output)
            test_loss.append(loss_fn(model_output, target).item())
            preds = (model_output > 0.5).float()

            # Apply sigmoid activation to get probabilities for each class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            # if batch_idx % 200 == 0:

            #     print(f"testing loss: {np.average(test_loss)}")
            #     break

    overall_accuracy = accuracy_score(
        np.concatenate(all_labels), np.concatenate(all_preds)
    )
    F1_scoure = f1_score(
        np.concatenate(all_labels), np.concatenate(all_preds), average="macro"
    )

    precision = precision_score(
        np.concatenate(all_labels), np.concatenate(all_preds), average="macro"
    )
    recall = recall_score(
        np.concatenate(all_labels), np.concatenate(all_preds), average="macro"
    )
    print(f"overall_accuracy: {overall_accuracy}")
    print(f"f1_scoure: {F1_scoure}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")

    writer.add_scalar("testing accuracy", overall_accuracy, global_step=epoch)
    writer.add_scalar("testing f1_scoure", F1_scoure, global_step=epoch)
    writer.add_scalar("testing precision", precision, global_step=epoch)
    writer.add_scalar("testing recall", recall, global_step=epoch)

    writer.add_scalar("testing loss", np.average(test_loss), global_step=epoch)

    return F1_scoure, np.average(test_loss)


def train_model(model, train_loader,validat_loader, optimizer, loss_fn, epoch, writer):
    model.train()
    print(f"training epoch: {epoch}")
    for index, (data, target) in tqdm.tqdm(enumerate(train_loader)):
        # Limiting training data for faster epochs.
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        logs = model(data)
        logs = F.sigmoid(logs)
        loss = loss_fn(logs, target)
        loss.backward()
        optimizer.step()
        # if index % 200 == 0:
        #     print(f"training loss: {loss.item()}")
        # if batch_idx % 200 == 0:
        #     print(f"training loss: {loss.item()}")

    writer.add_scalar("training loss", loss.item(), global_step=epoch)
    writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], global_step=epoch)
# Evaluate the model on the validation dataset
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(validat_loader)):
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            model_output = model(data)
            model_output = F.sigmoid(model_output)
            test_loss.append(loss_fn(model_output, target).item())
            preds = (model_output > 0.5).float()

            # Apply sigmoid activation to get probabilities for each class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            # if batch_idx % 200 == 0:

            #     print(f"testing loss: {np.average(test_loss)}")
            #     break

    overall_accuracy = accuracy_score( np.concatenate(all_labels), np.concatenate(all_preds)) 
    F1_scoure = f1_score( np.concatenate(all_labels), np.concatenate(all_preds), average="macro")
    
    precision = precision_score(
        np.concatenate(all_labels), np.concatenate(all_preds), average="macro"
    )
    recall = recall_score(
        np.concatenate(all_labels), np.concatenate(all_preds), average="macro"
    )
    print(f"overall_accuracy: {overall_accuracy}")
    print(f"f1_scoure: {F1_scoure}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")

    writer.add_scalar("valid_loss", np.average(test_loss), global_step=epoch)
    writer.add_scalar("valid_accuracy", overall_accuracy, global_step=epoch)
    writer.add_scalar("validation f1_scoure", F1_scoure, global_step=epoch)
    writer.add_scalar("valid_precision", precision, global_step=epoch)
    writer.add_scalar("valid_recall", recall, global_step=epoch)

    return model

class ClassifierModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierModel, self).__init__()
        model = torchvision.models.resnet50(pretrained=False)
        weights = torch.load("base_model/model.pt")
        model.load_state_dict(
            weights, strict=False
        )  # Add strict=False to handle mismatched keys
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.resnet = model

    def forward(self, x):
        x = self.resnet(x)

        return x


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCHSIZE = 256
    EPOCHS = 1000
    train_loader, test_loader,valid_data = get_data()
    model = ClassifierModel(num_classes=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, weight_decay=0.00002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0)
    writer = SummaryWriter(log_dir=f"runs/classifier")
    for epoch in range(EPOCHS):
        model=train_model(model, train_loader,valid_data, optimizer, loss_fn, epoch, writer)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"s/model-{epoch}.pt")
        scheduler.step()
    test_model(model, test_loader, loss_fn, epoch, writer)

    writer.close()
