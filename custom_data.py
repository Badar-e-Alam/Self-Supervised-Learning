from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image, ImageFile
import os


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
        img_loc2 = os.path.join(
            self.main_dir, self.all_imgs[(idx + 1) % self.total_imgs]
        )
        image1 = Image.open(img_loc1).resize((200, 200))
        image2 = Image.open(img_loc2).resize((200, 200))
        tensor_image1 = self.transform(image1)
        tensor_image2 = self.transform(image2)
        return tensor_image1, tensor_image2
