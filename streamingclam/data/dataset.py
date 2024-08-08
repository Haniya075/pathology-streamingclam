import pyvips
import torch
import math

import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A

augmentations = A.Compose(
    [
        A.Flip(),
        A.HueSaturationValue(p=0.5),
        A.Rotate(),
    ],
)


class StreamingClassificationDataset(Dataset):
    def __init__(
        self,
        img_dir: Path | str,
        csv_file: str | pd.DataFrame,
        tile_size: int,
        img_size: int,
        read_level: int,
        transform: A.BaseCompose | None = None,
        mask_dir: Path | str | None = None,
        mask_suffix: str = "_tissue",
        variable_input_shapes: bool = False,
        tile_stride: int | None = None,
        network_output_stride: int = 1,
        filetype=".png",
    ):
        self.img_dir = Path(img_dir)
        self.filetype = filetype
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix

        self.read_level = read_level
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.network_output_stride = network_output_stride
        self.img_size = img_size

        self.variable_input_shapes = variable_input_shapes
        self.transform = transform

        if not isinstance(csv_file, pd.DataFrame):
            self.classification_frame = pd.read_csv(csv_file)
        else:
            self.classification_frame = csv_file

        # Will be populated in check_csv function
        self.data_paths = {"images": [], "masks": [], "labels": []}

        self.random_crop = A.RandomCrop(self.img_size, self.img_size, p=1.0)
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Directory {self.img_dir} not found or doesn't exist")
        self.check_csv()

        self.labels = self.data_paths["labels"]
        print("SELF.DATAPATHS: ", self.data_paths)

    def check_csv(self):
        """Check if entries in csv file exist"""

        included = {"images": [], "masks": [], "labels": []} if self.mask_dir else {"images": [], "labels": []}
        
        existing_files = {f.stem[:12]: f for f in self.img_dir.glob(f"*{self.filetype}")}
        #print("EXISTING FILES : ",existing_files,"************")
        
        for i in range(len(self)):
            images, label = self.get_img_path(i)
            # Files can be just images, but also image, mask
            for file in images:
                file_key = file.stem[:12]
                if file_key not in existing_files:
                    #print(f"WARNING {file} not found, excluded both image and mask (if present)!")
                    continue
                else:
                    print("FILEEEEEEEEEEEEE : ",file)

        included["images"].append(images[0])
        included["labels"].append(label)

            if self.mask_dir:
                included["masks"].append(images[1])

        self.data_paths = included

    def get_img_path(self, idx):
        img_fname = self.classification_frame.iloc[idx, 0]
        label = self.classification_frame.iloc[idx, 1]

        img_path = self.img_dir / Path(img_fname).with_suffix(self.filetype)

        if self.mask_dir:
            mask_path = self.mask_dir / Path(img_fname[:12] + self.mask_suffix).with_suffix(self.filetype)
            return [img_path, mask_path], label

        return [img_path], label

    def get_img_pairs(self, idx):
        images = {"image": None}

        img_fname = str(self.data_paths["images"][idx])
        print("get_img_pairs: ",img_fname)
        label = int(self.data_paths["labels"][idx])
        image = pyvips.Image.new_from_file(img_fname, page=self.read_level)
        images["image"] = image

        if self.mask_dir:
            mask_fname = str(self.data_paths["masks"][idx])
            mask = pyvips.Image.new_from_file(mask_fname)
            ratio = image.width / mask.width
            images["mask"] = mask.resize(ratio, kernel="nearest")  # Resize mask to img size

        return images, label, img_fname

    def __getitem__(self, idx):
        sample, label, img_fname = self.get_img_pairs(idx)

        if self.transform:
            sample = self.transform(**sample)

        pad_to_tile_size = sample["image"].width < self.tile_size or sample["image"].height < self.tile_size
        resize_op = self.get_resize_op(pad_to_tile_size=pad_to_tile_size)
        sample = resize_op(**sample)

        if sample["image"].width * sample["image"].height > self.img_size**2:
            sample = self.random_crop(**sample)

        if "mask" in sample.keys():
            new_height = math.ceil(sample["mask"].height / self.network_output_stride)
            vscale = new_height / sample["mask"].height

            new_width = math.ceil(sample["mask"].width / self.network_output_stride)
            hscale = new_width / sample["mask"].width

            sample["mask"] = sample["mask"].resize(hscale, vscale=vscale, kernel="nearest")

        to_tensor = A.Compose([A.ToTensor(transpose_mask=True)], is_check_shapes=False)
        sample = to_tensor(**sample)

        if "mask" in sample.keys():
            sample["mask"] = sample["mask"] >= 1

        sample["label"] = torch.tensor(label)
        sample["image_name"] = Path(img_fname).stem
        return sample

    def __len__(self):
        return len(self.classification_frame)

    def get_resize_op(self, pad_to_tile_size=False):
        if not self.variable_input_shapes:
            return A.Compose([A.CropOrPad(self.img_size, self.img_size, p=1.0)])

        if pad_to_tile_size:
            return A.Compose(
                [
                    A.PadIfNeeded(
                        min_width=self.tile_size,
                        min_height=self.tile_size,
                        value=[255, 255, 255],
                        mask_value=[0, 0, 0],
                    ),
                    A.PadIfNeeded(
                        min_height=None,
                        min_width=None,
                        pad_width_divisor=self.network_output_stride,
                        pad_height_divisor=self.network_output_stride,
                        value=[255, 255, 255],
                        mask_value=[0, 0, 0],
                    ),
                ]
            )

        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_width_divisor=self.tile_stride,
                    pad_height_divisor=self.tile_stride,
                    value=[255, 255, 255],
                    mask_value=[0, 0, 0],
                ),
            ]
        )


if __name__ == "__main__":
    root = Path("/data/pathology/projects/pathology-bigpicture-streamingclam")
    data_path = '/kaggle/input/croppedd/new_cropped_images-Copy'
    mask_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images_tissue_masks")
    csv_file = '/kaggle/input/datasetcsv/dataset.csv'

    dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(csv_file),
        tile_size=1600,
        img_size=1600,
        transform=augmentations,
        mask_dir=mask_path,
        mask_suffix="_tissue",
        variable_input_shapes=False,
        tile_stride=680,
        filetype=".png",
        read_level=5,
    )

    import matplotlib.pyplot as plt

    ds = iter(dataset)
    print("Creating dataset")
    sample, label, name = next(ds)
    print("retrieving image")
    image = sample["image"]
    mask = sample["mask"]
    print("image shape", image.shape)
    print("image dtype", image.dtype)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.show()
    print("mask shape", mask.shape)
    plt.imshow(mask.cpu().numpy())
    plt.show()
