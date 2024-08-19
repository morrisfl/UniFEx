import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class GLDv2(Dataset):
    def __init__(self, root, min_samples=3, max_samples=100, num_cls=None, transform=None, augmentation=None,
                 data_amount=None, offset=0):
        """Parameters:
            root (str): Path to the root directory of the datasets.
            min_samples (int, optional): Minimum number of samples per class to use.
            max_samples (int, optional): Maximum number of samples per class to use.
            num_cls (int, optional): Number of classes to use.
            transform (torchvision.transforms, optional): Resize, normalize and convert image to tensor.
            augmentation (torchvision.transforms, optional): Data augmentation.
            data_amount (int, optional): Number of data to use.
            offset (int, optional): Offset of class labels.
        """
        self.root = os.path.join(root, "google_landmark_recognition_2021")
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.num_cls = num_cls

        self.img_dir = os.path.join(self.root, "train")

        self.df = pd.read_csv(os.path.join(self.root, "train.csv"), nrows=data_amount, converters={"": str})
        self.num_classes, self.landmark_ids = self._sampling()

        ids_to_label = {class_id: idx + offset for idx, class_id in enumerate(self.landmark_ids)}
        self.df["landmark_id"] = self.df["landmark_id"].map(ids_to_label)

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["id"]
        label = row["landmark_id"]
        img_path = os.path.join(self.img_dir, img_name[0], img_name[1], img_name[2], f"{img_name}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.augmentation:
            img = self.augmentation(img)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)

    def _sampling(self):
        if self.min_samples:
            self.df = self.df.groupby("landmark_id").filter(lambda x: len(x) >= self.min_samples)
        if self.max_samples:
            for name, group in self.df.groupby("landmark_id"):
                if len(group) > self.max_samples:
                    self.df = self.df.drop(group.sample(len(group) - self.max_samples).index)

        landmark_ids = sorted(list(self.df["landmark_id"].unique()))

        if self.num_cls is not None:
            groups = self.df.groupby("landmark_id")
            while self.num_cls < len(landmark_ids):
                sample_id = random.choice(landmark_ids)
                landmark_ids.remove(sample_id)
                self.df = self.df.drop(index=groups.get_group(sample_id).index)

        return len(landmark_ids), landmark_ids


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = GLDv2(data_root, transform=img_transform, data_amount=1000)
    number_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
