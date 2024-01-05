"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from custom_data import Winding_Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from functools import partial
from optuna.logging import _get_library_root_logger

logger = _get_library_root_logger()

expriment_results = "classifier-results"
if not os.path.exists(expriment_results):
    os.mkdir(expriment_results)

current_time = datetime.now().strftime("%b%d_%H-%M-%S")
logs = os.path.join(expriment_results, current_time)
writer = SummaryWriter(log_dir=logs)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 256
CLASSES = 3
EPOCH = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 50
N_VALID_EXAMPLES = BATCHSIZE * 10


weight_path = "train_weights/lightly-barlow-weights.pt"
barlow_weights = torch.load(weight_path)
barlow_backbone = torchvision.models.resnet50(pretrained=False)
barlow_backbone = nn.Sequential(*list(barlow_backbone.children())[:-1])
barlow_backbone.load_state_dict(barlow_weights)
barlow_backbone.eval()
barlow_backbone.to(DEVICE)


def early_stopping(study, trail, early_stopping=10):
    current_trial = trail.number
    best_trial = study.best_trial.number
    should_stop = (current_trial - best_trial) > early_stopping
    if should_stop:
        print("early stopping detected")
        logger.debug("early stopping detected: %s", should_stop)
        study.stop()


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3, 5)
    layers = []

    in_features = 2048
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 32, 512)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.4)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


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

            # if batch_idx < N_VALID_EXAMPLES:
            #     break
    # import pdb; pdb.set_trace()

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

    return F1_scoure


def objective(trial):
    # Generate the model.
    print("trial number: ", trial.number)
    writer = SummaryWriter(log_dir=f"optuna-exp/trial_{trial.number}")

    classification_criterion = nn.BCELoss()
    model = define_model(trial).to(DEVICE)
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    accuracy = []
    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_data()
    for epoch in range(EPOCH):
        train_model(
            model, train_loader, optimizer, classification_criterion, epoch, writer
        )
        acc = test_model(model, valid_loader, classification_criterion, epoch, writer)
        trial.report(acc, epoch)
        accuracy.append(acc)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print("pruning trial: ", trial.number)
            raise optuna.exceptions.TrialPruned()

    return np.average(accuracy)


if __name__ == "__main__":
    study_name = "classifier-study"
    storage_name = "sqlite:///classifier.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    early_stopping_callback = partial(early_stopping, early_stopping=20)

    study.optimize(objective, callbacks=[partial(early_stopping_callback)])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
