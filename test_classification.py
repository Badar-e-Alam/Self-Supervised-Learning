import torch

import torch.nn as nn   
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import torch.nn.functional as F
import torchvision 
from custom_data import Winding_Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 32
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
    
class ClassifierModel_base(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierModel_base, self).__init__()
        model = torchvision.models.resnet50(pretrained=False)
        weights = torch.load("classifier_head/base_model/base2/model-200.pt",map_location=DEVICE)
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
def get_data():
    # Load FashionMNIST dataset.
    dir_path = "/scratch/mrvl005h/Image_data"
    # train_dataset = Winding_Dataset(
    #     csv_file="data-csv/train.csv", root_dir=dir_path
    # )
    # valid_dataset = Winding_Dataset(
    #     csv_file="data-csv/validation.csv", root_dir=dir_path
    # )
    test_dataset = Winding_Dataset(
        csv_file="data-csv/linear_winding_labels_image_level_test_set_v2.0.csv", root_dir=dir_path
    )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=BATCHSIZE, shuffle=True
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=BATCHSIZE, shuffle=True
    # )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCHSIZE, shuffle=True
    )
    print("data loaded")
    return  test_loader
def test_model_base(model, test_loader, loss_fn, epoch, writer):
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

def test_model(model, feature_extractor,test_loader, loss_fn, epoch, writer):
    print(f"testing epoch: {epoch}")

    model.eval()
    feature_extractor.eval()
    model=model.to(DEVICE)  
    feature_extractor=feature_extractor.to(DEVICE)
    all_preds = []
    all_labels = []
    test_loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_loader)):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            barlow_features = feature_extractor(data)
            barlow_features = barlow_features.view(barlow_features.size(0), -1)
            model_output = model(barlow_features)
            test_loss.append(loss_fn(model_output, target).item())
            preds = (model_output > 0.5).float()

            # Apply sigmoid activation to get probabilities for each class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())

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

    # writer.add_scalar("testing accuracy", overall_accuracy, global_step=epoch)
    # writer.add_scalar("testing f1_scoure", F1_scoure, global_step=epoch)
    # writer.add_scalar("testing precision", precision, global_step=epoch)
    # writer.add_scalar("testing recall", recall, global_step=epoch)

    # writer.add_scalar("testing loss", np.average(test_loss), global_step=epoch)
    return F1_scoure, np.average(test_loss)

if __name__=="__main__":
    test_loader=get_data()
    print("code running on: ",DEVICE)
    model=ClassifierModel(num_features=2048,num_classes=3).to(DEVICE)
    # model=ClassifierModel_base(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load("/home/woody/rzku/mrvl005h/output1/dino_classification_big/Self-Supervised-Learning/classification_results/model/model_150.pt",map_location=DEVICE))
    dino_backbone = torch.load("train_weights/dino_backbone.pt",map_location=DEVICE)
    resnet=torchvision.models.resnet50()
    resnet=nn.Sequential(*list(resnet.children())[:-1])
    resnet.load_state_dict(dino_backbone)
    resnet.eval()
    print(f"Length of test_loader: {len(test_loader)}")
    loss=nn.BCEWithLogitsLoss()
    # loss=nn.CrossEntropyLoss()
    test_model(model,resnet,test_loader,loss,0,None)
    # test_model_base(model, test_loader, loss, 0,None)

