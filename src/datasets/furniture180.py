import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class Furniture180(Dataset):
    def __init__(self, root, transform=None, augmentation=None, offset=0):
        """Parameters:
            root (str): Path to the root directory of the datasets.
            transform (torchvision.transforms, optional): Resize, normalize and convert image to tensor.
            augmentation (torchvision.transforms, optional): Data augmentation.
            offset (int, optional): Offset of class labels.
        """
        self.root = os.path.join(root, "furniture_180")

        ann_path = os.path.join(self.root, "furniture180_train.csv")
        self.df = pd.read_csv(ann_path)

        self.class_ids = sorted(list(self.df["label"].unique()))
        self.num_classes = len(self.class_ids)

        id_to_label = {label_group: idx + offset for idx, label_group in enumerate(self.class_ids)}
        self.df["label"] = self.df["label"].map(id_to_label)

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        label = row["label"]
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

        if self.augmentation:
            img = self.augmentation(img)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = Furniture180(data_root, transform=img_transform)
    num_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
