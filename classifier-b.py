
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from custom_data import Winding_Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
BATCHSIZE = 256
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
barlow_backbone = torchvision.models.resnet50(weights=None)
barlow_backbone = nn.Sequential(*list(barlow_backbone.children())[:-1])
barlow_backbone.load_state_dict(torch.load("train_weights/Ligthly_trained.pt"))
barlow_backbone.eval()
barlow_backbone.to(DEVICE)
writer=SummaryWriter("classifier-results")


def test_model(model, test_loader, loss_fn, epoch, writer):
    print(f"testing epoch: {epoch}")
    
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            barlow_features = barlow_backbone(data)
            barlow_features = barlow_features.view(barlow_features.size(0), -1)
            model_output = model(barlow_features)
            test_loss.append(loss_fn(model_output, target).item())
            preds=(model_output>0.5).float()

            # Apply sigmoid activation to get probabilities for each class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())
            if batch_idx % 200 == 0:
                
                print(f"testing loss: {np.average(test_loss)}")

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

    return F1_scoure,np.average(test_loss)


def get_data():
    # Load FashionMNIST dataset.
    dir_path = "/scratch/mrvl005h/Image_data/"
    train_dataset = Winding_Dataset(csv_file="data-csv/training_set.csv", root_dir=dir_path)
    valid_dataset = Winding_Dataset(csv_file="data-csv/validation_set.csv", root_dir=dir_path)
    test_dataset = Winding_Dataset(
        csv_file="data-csv/test_set_representative.csv", root_dir=dir_path
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
    return train_loader, test_loader


def train_model(model, train_loader, optimizer, loss_fn, epoch, writer):
    model.train()
    print(f"training epoch: {epoch}")
    for batch_idx, (data, target) in enumerate(train_loader):
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
        if batch_idx % 200 == 0:
            print(f"training loss: {loss.item()}")
    writer.add_scalar("training loss", loss.item(), global_step=epoch)
    return model


class ClassifierModel(nn.Module):
    def __init__(self, num_features, num_classes):
            super(ClassifierModel, self).__init__()
            self.fc = nn.Sequential(
                    nn.Linear(num_features, 431),
                    nn.ReLU(),
                   # nn.Dropout(0.2979960641897257),
                    # nn.Dropout(dropout_rate),
                    nn.Linear(431, num_classes),
                    # nn.Softmax(dim=1) #because the problem is multiclass classification (4 classes) are equal to 1 
                    nn.Sigmoid(),#mutile label
                )

    def forward(self, x):
            x = self.fc(x)
            return x

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def main():
    train_loader, test_loader = get_data()
    model = ClassifierModel(2048, 3)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0019932144847006786)
    loss_fn = nn.BCELoss()
    early_stopping = EarlyStopping(patience=20, verbose=True)
    model_path="classifier-results/model/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for epoch in range(100, 1000):
        print(f"epoch: {epoch}")
        model = train_model(model, train_loader, optimizer, loss_fn, epoch, writer)
        F1_scoure,test_loss= test_model(model, test_loader, loss_fn, epoch, writer)
        early_stopping(test_loss, model)
        if epoch%100==0:
            model_file_path = os.path.join(model_path, f"model_{epoch}.pt")
            torch.save(model.state_dict(), model_file_path)#
            
        print(f"epoch: {epoch}, F1_scoure: {F1_scoure}")
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()