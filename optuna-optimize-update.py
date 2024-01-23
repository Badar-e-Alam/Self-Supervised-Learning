"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import tqdm
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



weight_path = "train_weights/dino_backbone.pt"
barlow_weights = torch.load(weight_path)
dino_backbackbone = torchvision.models.resnet50(pretrained=False)
dino_backbackbone = nn.Sequential(*list(dino_backbackbone.children())[:-1])
dino_backbackbone.load_state_dict(barlow_weights)
dino_backbackbone.eval()
dino_backbackbone.to(DEVICE)


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
            barlow_features = dino_backbackbone(data)
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
            barlow_features = dino_backbackbone(data)
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

    return np.average(valid_loss)


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
            barlow_features = dino_backbackbone(data)
            barlow_features = barlow_features.view(barlow_features.size(0), -1)
            model_output = model(barlow_features)
            test_loss.append(loss_fn(model_output, target).item())
            preds = (model_output > 0.5).float()

            # Apply sigmoid activation to get probabilities for each class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())
         

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
    # classification_criterion = nn.BCELoss()
    classification_criterion = nn.BCELoss()
    

    model = define_model(trial).to(DEVICE)
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    factor=trial.suggest_float("factor", 0.1, 0.9)
    patience=trial.suggest_int("patience", 1, 10)

    scheduler_name = trial.suggest_categorical("scheduler", ["ReduceLROnPlateau", "CosineAnnealingLR"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    if scheduler_name == "ReduceLROnPlateau":
        factor = trial.suggest_float("factor", 0.1, 0.9)
        patience = trial.suggest_int("patience", 10, 50)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor)
    elif scheduler_name == "CosineAnnealingLR":
        T_max = trial.suggest_int("T_max", 50, 1000)
        eta_min = trial.suggest_float("eta_min", 0, 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    accuracy = []
        # Get the FashionMNIST dataset.
    train_loader, valid_loader,test_loader = get_data()
    for epoch in range(EPOCH):
        val_loss=train_model(model, train_loader, valid_loader, optimizer, classification_criterion, epoch, writer)
        if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        acc = test_model(model,test_loader, classification_criterion, epoch, writer)
        trial.report(acc, epoch)
        accuracy.append(acc)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print("pruning trial: ", trial.number)
            raise optuna.exceptions.TrialPruned()

    return np.average(accuracy)


if __name__ == "__main__":
    study_name = "windings-classifier"
    storage_name = "sqlite:///classifier.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    early_stopping_callback = partial(early_stopping, early_stopping=20)

    study.optimize(objective, n_trials=100)

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
