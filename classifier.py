import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import tqdm
from custom_data import Winding_Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from sklearn.metrics import classification_report
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix,
    accuracy_score,
)
import numpy as np
import torch


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


def train_classifier(small_train=True, barlow=True):
    batch_size = 256
    erlay_stop = EarlyStopping(patience=10)

    # learning_rate = 0.0004029263492566185
    # momentum_term = 0.9022005260755215
    dropout_rate = 0.035579632226975244

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("classifier", current_time)
    writer = SummaryWriter(log_dir=log_dir)

    def topk_error(preds, labels, k):
        topk_preds = preds.topk(k, dim=1)[1]
        correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        error = 1 - correct_k.div_(labels.size(0))
        return error.item()

    class ClassifierModel(nn.Module):
        def __init__(self, num_features, num_classes):
            super(ClassifierModel, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(num_features, 500),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                # nn.Dropout(dropout_rate),
                nn.Linear(500, num_classes),
                # nn.Softmax(dim=1) #because the problem is multiclass classification (4 classes) are equal to 1
                nn.Sigmoid(),  # mutile label
            )

        def forward(self, x):
            x = self.fc(x)
            return x

        def forward(self, x):
            x = self.fc(x)
            return x

    barlow = barlow
    small_train = small_train
    if small_train:
        data_path = "/home/vault/rzku/mrvl005h/data/Image_data/single_class/"
    else:
        data_path = "/scratch/mrvl005h/Image_data/single_class"
    if barlow:
        model_path = "barlowtwins_weights.pt"
        barlow_weight = torch.load(model_path)
        barlow_backbone = torchvision.models.resnet50(pretrained=False)
        barlow_backbone = nn.Sequential(*list(barlow_backbone.children())[:-1])
    else:
        model_path = "meta_checkpoint.pt"
        barlow_weight = torch.load(model_path)
        barlow_backbone = torchvision.models.resnet50(pretrained=False)
        barlow_backbone.fc = nn.Identity()

    barlow_backbone.load_state_dict(barlow_weight)
    barlow_backbone.eval()
    classifier_model = ClassifierModel(num_features=2048, num_classes=3)
    ##
    classification_criterion = nn.BCELoss()
    train_dataset = Winding_Dataset("training_set.csv", data_path)
    test_dataset = Winding_Dataset("test_set_representative.csv", data_path)
    validation_dataset = Winding_Dataset("validation_set.csv", data_path)
    # test_size = int(0.2 * len(dataset))
    # train_size = len(dataset) - test_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    # Define your optimizer for the classifier head (e.g., Adam, SGD, etc.)#
    classification_optimizer = optim.Adam(classifier_model.parameters())
    # classification_optimizer = optim.RMSprop(classifier_model.parameters(),lr=learning_rate, momentum=momentum_term)
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    barlow_backbone.to(device)
    classifier_model.to(device)
    # Assuming you have a dataloader for your dataset
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        print(f"Epoch {epoch + 1}")
        running_loss = 0.0
        for index, data in tqdm.tqdm(enumerate(train_loader, 0)):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero the parameter gradients
            classification_optimizer.zero_grad()
            # Extract features from the pre-trained Barlow Twins model
            with torch.no_grad():
                barlow_features = barlow_backbone(inputs)
                barlow_features = torch.flatten(barlow_features, 1)
            # Forward pass through the classifier head
            classification_outputs = classifier_model(barlow_features)
            classification_outputs = classification_outputs.squeeze()
            # Compute classification loss
            classification_loss = classification_criterion(
                classification_outputs, labels
            )

            # Backward pass and optimization
            classification_loss.backward()
            classification_optimizer.step()

            # Print statistics
            running_loss += classification_loss
            if index % 200 == 199:  # Print every 2000 mini-batches
                print(f"[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0
                break

        correct = 0
        test_loss = 0
        total_samples = len(test_loader)
        # for data in test_loader:
        #     inputs, labels = data
        #     with torch.no_grad():
        #         barlow_features = barlow_backbone(inputs)
        #         barlow_features = torch.flatten(barlow_features, 1)
        #     classification_outputs = classifier_model(barlow_features)
        #     _, predicted = torch.max(classification_outputs.data, 1)
        #     import pdb; pdb.set_trace()
        #     total += labels.size(0)

        # all_predictions=[]
        # all_labels=[]
        # with torch.no_grad():
        #     for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_loader)):
        #         data, target = data.to(device), target.to(device)
        #         barlow_features = barlow_backbone(data)
        #         barlow_features = torch.flatten(barlow_features, 1)
        #         classification_outputs = classifier_model(barlow_features)
        #         test_loss += classification_criterion(classification_outputs.squeeze(), target).item()
        #         predicted_classes = torch.round(classification_outputs.squeeze())  # Round to get binary predictions

        #         all_predictions.extend(predicted_classes.cpu().numpy())
        #         all_labels.extend(target.cpu().numpy())  # Assuming target is already binary

        # # Calculate metrics after processing all batches
        # overall_accuracy = accuracy_score(all_labels, all_predictions)
        # overall_precision = precision_score(all_labels, all_predictions, average='binary')  # Change to 'binary'
        # overall_recall = recall_score(all_labels, all_predictions, average='binary')  # Change to 'binary'
        # overall_f1 = f1_score(all_labels, all_predictions, average='binary')  # Change to 'binary'
        # erlay_stop(overall_accuracy,classifier_model)
        ###mutlipclass classification
        all_predictions = []
        all_labels = []
        test_loss = []
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_loader)):
                data, target = data.to(device), target.to(device)
                barlow_features = barlow_backbone(data)
                barlow_features = torch.flatten(barlow_features, 1)
                classification_outputs = classifier_model(barlow_features)
                test_loss.append(
                    classification_criterion(
                        classification_outputs.squeeze(), target
                    ).item()
                )
                _, predicted_classes = torch.max(classification_outputs, 1)

                # Convert one-hot encoded labels to index tensor
                true_classes = torch.argmax(target, dim=1)
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_labels.extend(true_classes.cpu().numpy())
                # import pdb; pdb.set_trace()
                # # Compare predictions with ground truth
                # correct_predictions = (predicted_classes == true_classes).sum().item()

                # # Calculate accuracy
                # correct += correct_predictions / len(target)
                # # Assuming target is a 2D tensor, squeeze it to make it 1D if needed

                # # Compare predictions with ground truth
                # correct_predictions = (predicted_classes == target).sum().item()
                # correct += correct_predictions

            # Calculate accuracy after processing all batches

        overall_accuracy = accuracy_score(all_labels, all_predictions)
        overall_precision = precision_score(
            all_labels, all_predictions, average="micro"
        )
        overall_recall = recall_score(all_labels, all_predictions, average="micro")
        overall_f1 = f1_score(all_labels, all_predictions, average="micro")
        if barlow:
            model_path = "classifier_weights_barlow.pt"
            torch.save(classifier_model.state_dict(), model_path)
        else:
            model_path = "classifier_weights_meta.pt"
            torch.save(classifier_model.state_dict(), model_path)
        # Evaluate on the test set
        # Print the results
        print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")

        writer.add_scalar("train_loss", classification_loss, epoch)
        writer.add_scalar("Precision", overall_precision, epoch)
        writer.add_scalar("Recall", overall_recall, epoch)
        writer.add_scalar("F1", overall_f1, epoch)

        writer.add_scalar(
            "Learning_rate", classification_optimizer.param_groups[0]["lr"], epoch
        )
        writer.add_scalar("Loss/test", np.average(test_loss), epoch)
        writer.add_scalar("Accuracy", 100 * overall_accuracy, epoch)

    print("Finished Fine-tuning")


if __name__ == "__main__":
    train_classifier(small_train=False, barlow=True)
    train_classifier(small_train=False, barlow=False)
