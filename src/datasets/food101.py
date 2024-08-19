import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class Food101(Dataset):
    def __init__(self, root, samples_per_cls=None, transform=None, augmentation=None, offset=0):
        """Parameters:
            root (str): Path to the root directory of the datasets.
            samples_per_cls (int, optional): Number of samples per class to use.
            transform (torchvision.transforms, optional): Resize, normalize and convert image to tensor.
            augmentation (torchvision.transforms, optional): Data augmentation.
            offset (int, optional): Offset of class labels.
        """
        self.root = os.path.join(root, "food-101")
        self.img_dir = os.path.join(self.root, "images")
        self.samples_per_cls = samples_per_cls

        ann_path = os.path.join(self.root, "meta", "test.json")
        with open(ann_path) as f:
            self.cls_to_samples = json.load(f)

        if self.samples_per_cls is not None:
            self._sampling()

        self.class_ids = sorted(list(self.cls_to_samples.keys()))
        self.num_classes = len(self.class_ids)
        id_to_label = {cat_id: idx + offset for idx, cat_id in enumerate(self.class_ids)}

        self.samples = []
        for cls_name, samples in self.cls_to_samples.items():
            for sample in samples:
                self.samples.append((sample, id_to_label[cls_name]))

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, f"{img_name}.jpg")).convert("RGB")

        if self.augmentation:
            img = self.augmentation(img)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def _sampling(self):
        for cls_name, samples in self.cls_to_samples.items():
            if len(samples) > self.samples_per_cls:
                self.cls_to_samples[cls_name] = random.sample(samples, self.samples_per_cls)


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = Food101(data_root, transform=img_transform)
    number_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
