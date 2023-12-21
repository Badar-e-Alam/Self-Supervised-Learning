from ffcv.loader import Loader
from ffcv.transforms import ToTensor, ToTorchImage
from ffcv.fields.decoders import IntDecoder
from lightly.transforms.byol_transform import (
    BYOLView1Transform,
    BYOLView2Transform,
)
# Define your transformations here
view_1_transform = BYOLView1Transform(input_size=32, gaussian_blur=0.0)
view_2_transform = BYOLView2Transform(input_size=32, gaussian_blur=0.0)

# Data decoding and augmentation
image_pipeline = [view_1_transform, view_2_transform, ToTensor(), ToTorchImage()]

# Pipeline for each data field
pipelines = {"image": image_pipeline,}

# Use the Loader from ffcv
train_dataloader = Loader("dataset.beton", batch_size=256, num_workers=8)
import pdb; pdb.set_trace()
for batch in train_dataloader:
    import pdb; pdb.set_trace()