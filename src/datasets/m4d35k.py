import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class M4D35k(Dataset):
    def __init__(self, root, transform=None, augmentation=None, data_amount=None, offset=0):
        """Parameters:
            root (str): Path to the root directory of the datasets.
            transform (torchvision.transforms, optional): Resize, normalize and convert image to tensor.
            augmentation (torchvision.transforms, optional): Data augmentation.
            data_amount (int, optional): Number of data to use.
            offset (int, optional): Offset of class labels.
        """
        self.root = root
        self.df = pd.read_csv(os.path.join(self.root, "m4d-35k_train.csv"), nrows=data_amount)
        self.class_ids = sorted(list(self.df["label"].unique()))
        self.num_classes = len(self.class_ids)

        id_to_label = {label: idx + offset for idx, label in enumerate(self.class_ids)}
        self.df["label"] = self.df["label"].map(id_to_label)

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["img_path"])
        img = Image.open(img_path).convert("RGB")
        label = row["label"]

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

    dataset = M4D35k(data_root, transform=img_transform)
    number_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        pass
