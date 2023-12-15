# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import torch
import torchvision
from torch import nn

import tqdm
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
import torch.optim as optim
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
import math
from PIL import Image
from custom_data import Image_dataset



class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

resume = False
batch_size = 256
big_train = False
resnet = torchvision.models.resnet50()
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
backbone = nn.Sequential(*list(resnet.children())[:-1])
if resume:
    print("Resume training")
    weight_path = "model.pt"
    checkpoint=torch.load(weight_path)
    backbone.load_state_dict(checkpoint)
        
model = BarlowTwins(backbone)
writer = SummaryWriter("runs/barlowtwins")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")
model.to(device)

# BarlowTwins uses BYOL augmentations.
# We disable resizing and gaussian blur for cifar10.
transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=200, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=200, gaussian_blur=0.0),
)



def adjust_learning_rate(epoch, optimizer, loader, step):
    max_steps = epoch * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]["lr"] = lr * 0.2
    optimizer.param_groups[1]["lr"] = lr * 0.048

def cosine_lr_scheduler(optimizer, warmup_iters, num_iters, lr_max, lr_min):
    """Cosine learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to use.
        warmup_iters (int): The number of warmup iterations.
        num_iters (int): The total number of iterations.
        lr_max (float): The maximum learning rate.
        lr_min (float): The minimum learning rate.
    """

    def lr_lambda(global_step: int):
        if global_step < warmup_iters:
            return float(global_step) / float(max(1.0, warmup_iters))
        return 0.5 * (1.0 + math.cos(math.pi * (global_step - warmup_iters) / float(max(1.0, num_iters - warmup_iters))))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Create a learning rate scheduler
if big_train:
    Customedata = Image_dataset(main_dir="/scratch/mrvl005h/Image_data/",transform=transform)
else:
    Customedata = Image_dataset(main_dir="/home/vault/rzku/mrvl005h/data/Image_data/",transform=transform)

print("loading data from: ", Customedata.main_dir)
train_data_loader = torch.utils.data.DataLoader(
    Customedata,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)
optimizer = torch.optim.Adam(model.parameters())
scheduler = cosine_lr_scheduler(optimizer, warmup_iters=100, num_iters=10000, lr_max=1e-3, lr_min=1e-5)

criterion = BarlowTwinsLoss()
print("Starting Training")
model.train()
for epoch in tqdm.tqdm(range(1000)):
    avg_loss = 0.0
    total_loss = 0.0
    print(f"epoch: {epoch:>02}")
    for index, batch in tqdm.tqdm(enumerate(train_data_loader)):
        import pdb; pdb.set_trace()
        x0, x1 = batch
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        #adjust_learning_rate(epoch, optimizer, train_data_loader, index)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    avg_loss = total_loss / len(train_data_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

    print(f"epoch: {epoch:>02}  avg_loss: {avg_loss:.2f}")
    if epoch % 50 == 0:
        print("Saving model")
        torch.save(model.backbone.state_dict(), f"runs/barlowtwins/model.pt")
