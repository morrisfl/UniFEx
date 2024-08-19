import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class HuMPersonalisedFashion(Dataset):
    def __init__(self, root, min_samples=3, max_samples=None, num_cls=None, transform=None, augmentation=None,
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
        self.root = os.path.join(root, "hm_personalized_fashion")
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.num_cls = num_cls

        self.img_path = os.path.join(self.root, "images")

        self.df = pd.read_csv(os.path.join(self.root, "articles.csv"), nrows=data_amount, dtype=str)
        self.num_classes, self.class_ids = self._sampling()

        id_to_label = {cls_id: idx + offset for idx, cls_id in enumerate(self.class_ids)}
        self.df["product_code"] = self.df["product_code"].map(id_to_label)

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["article_id"] + ".jpg"
        img_path = os.path.join(self.img_path, img_name[:3], img_name)
        img = Image.open(img_path).convert("RGB")
        label = row["product_code"]

        if self.augmentation:
            img = self.augmentation(img)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)

    def _sampling(self):
        if self.min_samples:
            self.df = self.df.groupby("product_code").filter(lambda x: len(x) >= self.min_samples)
        if self.max_samples:
            for name, group in self.df.groupby("product_code"):
                if len(group) > self.max_samples:
                    self.df = self.df.drop(group.sample(len(group) - self.max_samples).index)

        class_ids = sorted(list(self.df["product_code"].unique()))

        if self.num_cls is not None:
            groups = self.df.groupby("product_code")
            while self.num_cls < len(class_ids):
                sample_id = random.choice(class_ids)
                class_ids.remove(sample_id)
                self.df = self.df.drop(index=groups.get_group(sample_id).index)

        return len(class_ids), class_ids


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = HuMPersonalisedFashion(data_root, transform=img_transform)
    number_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
