import torch
import numpy as np
import torchvision
import torch.nn as nn
from custom_data import Winding_Dataset
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tqdm
import time

BATCHSIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Code is executed on:", DEVICE)
weight = "train_weights/dino_backbone.pt"
barlow_weight = torch.load(weight,map_location=DEVICE)
resnet = torchvision.models.resnet50()
barlow_backbone = nn.Sequential(*list(resnet.children())[:-1])
barlow_backbone.load_state_dict(barlow_weight)
# barlow_backbone.load_state_dict(torch.load("train_weights/lightly-barlow-weights.pt"))
barlow_backbone.to(DEVICE)
barlow_backbone.eval()
writer = SummaryWriter("classification_results/")


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
            barlow_time = time.time()
            barlow_features = barlow_backbone(data)
            barlow_features = barlow_features.view(barlow_features.size(0), -1)
            barlow_time = time.time()
            model_output = model(barlow_features)
            test_loss.append(loss_fn(model_output, target).item())
            preds = (model_output > 0.3).float()

            # Apply sigmoid activation to get probabilities for each class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            '''if batch_idx % 200 == 0:

                print(f"testing loss: {np.average(test_loss)}")
                break'''

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
    # print(f"overall_accuracy: {overall_accuracy}")
    # print(f"f1_scoure: {F1_scoure}")
    # print(f"precision: {precision}")
    # print(f"recall: {recall}")

    writer.add_scalar("testing accuracy", overall_accuracy, global_step=epoch)
    writer.add_scalar("testing f1_scoure", F1_scoure, global_step=epoch)
    writer.add_scalar("testing precision", precision, global_step=epoch)
    writer.add_scalar("testing recall", recall, global_step=epoch)

    writer.add_scalar("testing loss", np.average(test_loss), global_step=epoch)


def get_data():
    # Load FashionMNIST dataset.
    dir_path = "/scratch/mrvl005h/Image_data"
    train_dataset = Winding_Dataset(
        csv_file="data-csv/train.csv", root_dir=dir_path
    )
    valid_dataset = Winding_Dataset(
        csv_file="data-csv/validation.csv", root_dir=dir_path
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



def train_model(model, train_loader, valid_loader, optimizer, loss_fn, epoch, writer):
    print(f"training epoch: {epoch}")
    # for epoch in range(num_epochs):
        # Train the model on the training dataset
    model.train()
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):
        # Limiting training data for faster epochs.
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            barlow_features = barlow_backbone(data)
            barlow_features = barlow_features.view(barlow_features.size(0), -1)
        classifer_features = model(barlow_features)
        loss = loss_fn(classifer_features, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
        #     print(f"training loss: {loss.item()}")
        #     break
        writer.add_scalar("training loss", loss.item(), global_step=epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

    # Evaluate the model on the validation dataset
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        valid_loss = []
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(valid_loader)):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            barlow_features = barlow_backbone(data)
            barlow_features = barlow_features.view(barlow_features.size(0), -1)
            model_output = model(barlow_features)
            valid_loss.append(loss_fn(model_output, target).item())
            preds = (model_output > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            # if batch_idx % 10 == 0:
            #     print(f"validation loss: {np.average(valid_loss)}")
            #     break
        overall_accuracy = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
        F1_scoure = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average="macro")
        precision = precision_score(np.concatenate(all_labels), np.concatenate(all_preds), average="macro")
        recall = recall_score(np.concatenate(all_labels), np.concatenate(all_preds), average="macro")
        
        print(f"epoch: {epoch}")
        print(f"overall_accuracy: {overall_accuracy}")
        print(f"f1_scoure: {F1_scoure}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        writer.add_scalar("valid_loss", np.average(valid_loss), global_step=epoch)
        writer.add_scalar("valid_accuracy", overall_accuracy, global_step=epoch)
        writer.add_scalar("valid f1_scoure", F1_scoure, global_step=epoch)
        writer.add_scalar("valid_precision", precision, global_step=epoch)
        writer.add_scalar("valid_recall", recall, global_step=epoch)

    return model,np.average(valid_loss)

"""
chaning the model as per the optuna 
{'n_layers': 1, 'n_units_l0': 97, 'dropout_l0': 0.1957759240034611, 'optimizer': 'Adam', 'factor': 0.8061593655690279, 'patience': 8, 'scheduler': 'ReduceLROnPlateau', 'lr': 0.00268235679789218}
"""

class ClassifierModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ClassifierModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1957759240034611),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 431),
            nn.LeakyReLU(),
            nn.Dropout(0.1957759240034611),
            nn.Linear(431, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def main():
    print("Code is executed on:", DEVICE)
    print("Loading data")
    train_loader, test_loader,validation_data = get_data()
    model = ClassifierModel(2048, 3)

    # weight_path = "train_weights/multilbl_epoch_300.pt"
    # model.load_state_dict(torch.load(weight_path))
    model.to(DEVICE)
    """'optimizer': 'Adam', 'factor': 0.8061593655690279, 'patience': 8, 'scheduler': 'ReduceLROnPlateau', 'lr': 0.00268235679789218"""
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00268235679789218)  # lr=0.0019932144847006786
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.8061593655690279) # start decay lr if val_loss not decrease
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=0.00002)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
    loss_fn = nn.BCELoss()
    model_path = "classification_results/model/"

    # early_stopping = EarlyStopping(patience=20,path=model_path, verbose=True)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for epoch in tqdm.tqdm(range(0, 200)):
        print(f"epoch: {epoch}")
        model,valid_loss = train_model(model, train_loader, validation_data, optimizer, loss_fn, epoch, writer)
        scheduler.step(valid_loss)
        # early_stopping(valid_loss, model)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
        if epoch % 50 == 0:
            model_file_path = os.path.join(model_path, f"model_{epoch}.pt")
            torch.save(model.state_dict(), model_file_path)  #

    test_model(model, test_loader, loss_fn, epoch, writer)
        # if best_val_acc is None:
        #     best_val_acc = F1_scoure
        # elif F1_scoure > best_val_acc:
        #     best_val_acc = F1_scoure

        # early_stopping(test_loss, model)
        # scheduler.step(F1_scoure)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
