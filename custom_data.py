from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import random
import numpy as np

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
def save_images_as_png(tensor_images, file_names):
            for i in range(len(tensor_images)):
                image = tensor_images[i].permute(1, 2, 0).numpy()
                plt.imshow(image)
                plt.axis('off')
                plt.savefig(file_names[i], bbox_inches='tight', pad_inches=0)
                plt.close()
class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            #transforms.RandomResizedCrop(150, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.Resize((200, 200)),
            #transforms.RandomResizedCrop(150, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        
        return y1, y2
       
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(PIL.ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return PIL.ImageOps.solarize(img)
        else:
            return img

class Image_dataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.transform=Transform()
        self.all_imgs = os.listdir(main_dir)
        self.total_imgs = len(self.all_imgs)

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

        img_loc1 = os.path.join(self.main_dir, self.all_imgs[idx])
        # img_loc2 = os.path.join(
        #     self.main_dir, self.all_imgs[(idx + 1) % self.total_imgs]
        # )
        x = PIL.Image.open(img_loc1).convert("L").resize((200, 200))
        # image2 = Image.open(img_loc2).resize((200, 200))
        image1,image2=self.transform(x)

        # tensor_image1 = self.transform(image1)
        # tensor_image2 = self.transform(image2)
        return image1, image2



class Winding_Dataset(Dataset):
    def __init__(self, csv_file,root_dir):
        self.data = pd.read_csv(csv_file)
        self.transform=transforms.ToTensor()
        self.main_dir = root_dir
        self.total_imgs = len(self.data)

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_loc1 = os.path.join(self.main_dir, self.data.iloc[idx, 0])
        x = PIL.Image.open(img_loc1).convert("RGB").resize((200, 200))
        column_names = self.data.columns[2:]  # Get the names of all columns starting from the third one
        labels = self.data.loc[idx, column_names]  # Select data using column names
        labels = torch.tensor(labels, dtype=torch.float32)
        image = self.transform(x)
        return image, labels
