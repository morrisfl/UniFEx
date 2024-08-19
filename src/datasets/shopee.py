import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class Shopee(Dataset):
    def __init__(self, root, min_samples=3, max_samples=None, transform=None, augmentation=None, data_amount=None,
                 offset=0):
        """Parameters:
            root (str): Path to the root directory of the datasets.
            min_samples (int, optional): Minimum number of samples per class to use.
            max_samples (int, optional): Maximum number of samples per class to use.
            transform (torchvision.transforms, optional): Resize, normalize and convert image to tensor.
            augmentation (torchvision.transforms, optional): Data augmentation.
            data_amount (int, optional): Number of data to use.
            offset (int, optional): Offset of class labels.
        """
        self.root = os.path.join(root, "shopee")
        self.min_samples = min_samples
        self.max_samples = max_samples

        self.img_dir = os.path.join(self.root, "train_images")
        self.df = pd.read_csv(os.path.join(self.root, "train.csv"), nrows=data_amount)
        self._sampling()

        self.class_ids = sorted(list(self.df["label_group"].unique()))
        self.num_classes = len(self.class_ids)

        id_to_label = {label_group: idx + offset for idx, label_group in enumerate(self.class_ids)}
        self.df["label_group"] = self.df["label_group"].map(id_to_label)

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        label = row["label_group"]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

        if self.augmentation:
            img = self.augmentation(img)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)

    def _sampling(self):
        if self.min_samples:
            self.df = self.df.groupby("label_group").filter(lambda x: len(x) >= self.min_samples)
        if self.max_samples:
            for name, group in self.df.groupby("label_group"):
                if len(group) > self.max_samples:
                    self.df = self.df.drop(group.sample(len(group) - self.max_samples).index)


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = Shopee(data_root, transform=img_transform)
    number_classes = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
