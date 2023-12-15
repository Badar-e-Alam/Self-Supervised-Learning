from ffcv.fields import RGBImageField,TorchTensorField
from ffcv.writer import DatasetWriter
from custom_data import Image_dataset
import torch



# Define the transform to normalize the data

# Create the dataset
dataset = Image_dataset(main_dir="/home/vault/rzku/mrvl005h/data/Image_data/")


# Create a data loader
final_path="dataset.beton"

writer = DatasetWriter(final_path, {
    'image': TorchTensorField(dtype=torch.float32, shape=(1, 200, 200)),
}, num_workers=8)
writer.from_indexed_dataset(dataset)


