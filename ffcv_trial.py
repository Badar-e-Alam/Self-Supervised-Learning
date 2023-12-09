import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import ffcv
from torchvision.transforms import ToTensor


class Image_dataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.all_imgs = os.listdir(main_dir)
        self.transform = ToTensor()
        self.total_imgs = len(self.all_imgs)

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        img_loc1 = os.path.join(self.main_dir, self.all_imgs[idx])
        image1 = Image.open(img_loc1).resize((224, 224))
        tensor_image1 = self.transform(image1)
        return tensor_image1


# Define the transform to normalize the data

# Create the dataset

dataset = Image_dataset(main_dir="/scratch/mrvl005h/data/")


# Create a data loader


def convert_dataset(dset, name):
    writer = ffcv.writer.DatasetWriter(
        name + ".beton", {"image": ffcv.fields.RGBImageField()}
    )
    writer.from_indexed_dataset(dset)


convert_dataset(dataset, "cifar_train")
