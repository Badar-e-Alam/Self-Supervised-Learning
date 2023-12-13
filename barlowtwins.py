# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import optuna
import torch
import torchvision
from torch import nn

import plotly.io as pio
import tqdm
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)

# import ffcv
# import ffcv.fields.decoders as decoders
# from ffcv.fields import RGBImageField, IntField
# from ffcv.writer import DatasetWriter
# from ffcv.fields import RGBImageField
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
from PIL import Image
import os
from custom_data import Image_dataset
import json


# from ffcv.writer import DatasetWriter
# from ffcv.fields import RGBImageField, IntField
# from torch.utils.data import Dataset, DataLoader
# from ffcv.loader import Loader
# from ffcv.transforms import ToTensor, ToDevice
# import ffcv
# import ffcv.fields as fields
# import ffcv.fields.decoders as decoders
# import ffcv.transforms as transforms
class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


resnet = torchvision.models.resnet50()
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = BarlowTwins(backbone)
writer = SummaryWriter("runs/barlowtwins")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")
model.to(device)

# BarlowTwins uses BYOL augmentations.
# We disable resizing and gaussian blur for cifar10.
transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
)
# output_file = 'dataset.beton'
# main_dir="data/"
# def write_dataset_to_beton(dataset, output_file):
#     writer = DatasetWriter(output_file, {
#         'image1': RGBImageField(),
#         'image2': RGBImageField(),
#         'label': IntField()
#     })
#     writer.from_indexed_dataset(dataset)
#     writer.close()

# if not os.path.exists(main_dir):
#     raise FileNotFoundError(f"The directory {main_dir} does not exist.")

# # Create an instance of your dataset
# dataset = Image_dataset(main_dir)
# CIFAR_MEAN = [0.485, 0.456, 0.406]
# CIFAR_STD = [0.229, 0.224, 0.225]

# def get_image_pipeline(train=True):
#     augmentation_pipeline = (
#         [
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomTranslate(padding=2),
#             transforms.Cutout(8, tuple(map(int, CIFAR_MEAN))),
#         ]
#         if train
#         else []
#     )

#     image_pipeline = (
#         [decoders.SimpleRGBImageDecoder()]
#         + augmentation_pipeline
#         + [
#             transforms.ToTensor(),
#             transforms.ToDevice(device, non_blocking=True),
#             transforms.ToTorchImage(),
#             transforms.Convert(torch.float32),
#             torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#         ]
#     )
#     return image_pipeline

# # Write the dataset to a binary file if the output file doesn't exist

# train_image_pipeline = get_image_pipeline(train=True)

# if not os.path.exists(output_file):
#     write_dataset_to_beton(dataset, output_file)

# loader = Loader(
#     output_file,
#     batch_size=32,
#     num_workers=4,
#     order=ffcv.loader.OrderOption.SEQUENTIAL,
#     drop_last=False,
# )

# dataset = Image_dataset(main_dir='/scratch/mrv1005h/data/')
# train_data = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform, train=True
# )
# # test_data = torchvision.datasets.CIFAR10(
# #     "datasets/cifar10", download=True, transform=transform, train=False
# # )
# #or create a dataset from a folder containing images or videos:
# #dataset = LightlyDataset("/scratch/mrv1005h/data/", transform=transform)
Customedata = Image_dataset(main_dir="/scratch/mrvl005h/data")


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [32,64,128,256])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    step_size = trial.suggest_categorical("step_size", [10, 20, 30, 40, 50])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=0.1
    )
    print(f"batch_size: {batch_size}")
    train_data_loader = torch.utils.data.DataLoader(
        Customedata,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    # test_data_loader = torch.utils.data.DataLoader(
    #     test_data,
    #     batch_size=50,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=0,
    # )

    criterion = BarlowTwinsLoss()
    print("Starting Training")
    avg_loss = 0.0
    # for epoch in tqdm.tqdm(range(100)):
    total_loss = 0.0
    #     print(f"epoch: {epoch:>02}")
    for index, batch in tqdm.tqdm(enumerate(train_data_loader)):
        x0, x1 = batch
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # if index % 100 == 0:
        # iter_num = index + 1
        # writer.add_scalar("Loss/train", round(loss.item(), 2), iter_num)
        # img_grid = make_grid(x0)
        # writer.add_image('Input Images/x0', img_grid, iter_num)
        # img_grid = make_grid(x1)
        # writer.add_image('Input Images/x1', img_grid, iter_num)
        # writer.add_histogram('Model outputs/z0', z0, iter_num)
        # writer.add_histogram('Model outputs/z1', z1, iter_num)

    avg_loss = total_loss / len(train_data_loader)

    torch.save(model.backbone.state_dict(), f"runs/barlowtwins/model_{0}.pt")
    return avg_loss.item()


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

important_fig = optuna.visualization.plot_param_importances(study)
pio.write_image(important_fig, 'param_importances.png')

intermediate = optuna.visualization.plot_intermediate_values(study)
pio.write_image(intermediate, 'intermediate_values.png')

important_fig = optuna.visualization.plot_param_importances(study)
mpl_fig = mpl_to_plotly(important_fig)
plt.savefig("param_importances1.png")

intermediate = optuna.visualization.plot_intermediate_values(study)
mpl_fig = mpl_to_plotly(intermediate)
plt.savefig("intermediate_values1.png")

# Save the best trial values to a JSON file

# Print the optimization results
print("Number of finished trials:", len(study.trials))

best_trials = []
# trials_df = study.trials_dataframe()

# # Convert the DataFrame to a JSON string
# json_string = trials_df.to_json(orient="records", lines=True)

# # Specify the file path where you want to save the JSON file
# json_file_path = "optuna_results.json"

# # Write the JSON string to the file
# with open(json_file_path, "w") as json_file:
#     json_file.write(json_string)


for trial in study.trials:
    best_trial_values = {"value": trial.value, "params": trial.params}
    best_trials.append(best_trial_values)

with open("optuna_logging.json", "w") as file:
    json.dump(best_trials, file)

print(
    f"optuna logging file saved to {file.name}")
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
