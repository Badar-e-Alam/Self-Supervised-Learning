import torch
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
        )  # get the next image
        image1 = Image.open(img_loc1).resize((200, 200))
        image2 = Image.open(img_loc2).resize((200, 200))
        tensor_image1 = self.transform(image1)
        tensor_image2 = self.transform(image2)
        return tensor_image1, tensor_image2


# import nvidia.dali.fn as fn
# import nvidia.dali.types as types
# from nvidia.dali.pipeline import Pipeline
# from nvidia.dali.plugin.pytorch import DALIGenericIterator
# from PIL import Image
# import os

# def check_images(main_dir):
#     all_imgs = os.listdir(main_dir)
#     total_imgs = len(all_imgs)
#     valid_imgs = 0
#     for img_file in all_imgs:
#         try:
#             img_path = os.path.join(main_dir, img_file)
#             with Image.open(img_path) as img:
#                 img.verify()
#                 valid_imgs += 1
#         except Exception as e:
#             print(f"Error processing {img_file}: {e}")
#     print(f"Total images: {total_imgs}")
#     print(f"Valid images: {valid_imgs}")

# class ImagePipeline(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir):
#         super(ImagePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
#         self.input = fn.external_source(device=device_id, name="DALI_INPUT")
#         self.decode = fn.image_decoder(device=device_id, output_type=types.RGB)
#         self.resize = fn.resize(device=device_id, resize_x=224, resize_y=224)
#         self.cmnp = fn.crop_mirror_normalize(device=device_id,
#                                               output_dtype=types.FLOAT,
#                                               output_layout=types.NCHW,
#                                               crop=(224, 224),
#                                               mean=[0.485, 0.456, 0.406],
#                                               std=[0.229, 0.224, 0.225])
#         self.coin = fn.coin_flip(probability=0.5)

#     def define_graph(self):
#         rng = self.coin()
#         self.jpegs = self.input()
#         images = self.decode(self.jpegs)
#         images = self.resize(images)
#         output = self.cmnp(images.gpu(), mirror=rng)
#         return (output, )

# class ImageDataset(DALIGenericIterator):
#     def __init__(self, main_dir, batch_size, num_threads, device_id):
#         self.all_imgs = os.listdir(main_dir)
#         self.total_imgs = len(self.all_imgs)
#         print(check_images(main_dir))
#         self.pipe = ImagePipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, data_dir=main_dir)
#         self.pipe.build()
#         super().__init__(self.pipe, ["DALI_INPUT"], batch_size, auto_reset=True, fill_last_batch=False)

# from nvidia.dali import pipeline_def
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types
# from nvidia.dali.plugin.pytorch import DALIGenericIterator

# class Image_dataset:
#     def __init__(self, main_dir, batch_size, num_threads, device_id):
#         self.main_dir = main_dir
#         self.all_imgs = os.listdir(main_dir)
#         self.total_imgs = len(self.all_imgs)
#         self.batch_size = batch_size
#         self.num_threads = num_threads
#         self.device_id = device_id

#         self.pipes = self.dali_pipeline(device_id=self.device_id, num_threads=self.num_threads, batch_size=self.batch_size)
#         self.pipes.build()
#         self.loader = DALIGenericIterator(self.pipes, ['data1', 'data2'], self.total_imgs)

#     @pipeline_def
#     def dali_pipeline(self, device_id, num_threads, batch_size):
#         images = fn.readers.file(file_root=self.main_dir, file_list=None, random_shuffle=True)
#         images = fn.decoders.image(images, device='mixed', output_type=types.RGB)
#         images = fn.resize(images, resize_x=224, resize_y=224)

#         # Apply your transformations here
#         # For example, to apply a random horizontal flip:
#         images1 = fn.flip(images, horizontal=True)
#         images2 = fn.flip(images, horizontal=True)

#         return {"data1": images1, "data2": images2}

#     def __len__(self):
#         return self.total_imgs // self.batch_size

#     def __iter__(self):
#         return self.loader
