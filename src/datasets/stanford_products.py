import json
import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class StanfordProducts(Dataset):
    def __init__(self, root, min_samples=3, max_samples=None, num_cls=None, transform=None, augmentation=None, offset=0):
        """Parameters:
            root (str): Path to the root directory of the datasets.
            min_samples (int, optional): Minimum number of samples per class to use.
            max_samples (int, optional): Maximum number of samples per class to use.
            num_cls (int, optional): Number of classes to use.
            transform (torchvision.transforms, optional): Resize, normalize and convert image to tensor.
            augmentation (torchvision.transforms, optional): Data augmentation.
            offset (int, optional): Offset of class labels.
        """
        self.root = os.path.join(root, "stanford_online_products")
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.num_cls = num_cls

        self.id_to_samples = self._sampling()
        self.class_ids = sorted(list(self.id_to_samples.keys()))
        self.num_classes = len(self.class_ids)
        self.id_to_label = {cls_id: idx + offset for idx, cls_id in enumerate(self.class_ids)}

        self.samples = []
        for label, samples in self.id_to_samples.items():
            for sample in samples:
                self.samples.append((sample, self.id_to_label[label]))

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.augmentation:
            img = self.augmentation(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def _sampling(self):
        txt_path = os.path.join(self.root, "Ebay_train.txt")
        txt_lines = open(txt_path, "r").readlines()

        id_to_samples = defaultdict(list)
        for i, line in enumerate(txt_lines):
            if i > 0:
                img_id, cls_id, super_cls_id, img_path = line.split(" ")
                img_path = img_path.split("\n")[0]
                id_to_samples[int(cls_id)].append(img_path)

        if self.min_samples or self.max_samples:
            discard_ids = []
            for id, samples in id_to_samples.items():
                if self.min_samples:
                    if len(samples) < self.min_samples:
                        discard_ids.append(id)
                if self.max_samples:
                    if len(samples) > self.max_samples:
                        id_to_samples[id] = random.sample(samples, self.max_samples)
            for ids in discard_ids:
                id_to_samples.pop(ids)

        if self.num_cls is not None:
            id_to_samples = dict(random.sample(id_to_samples.items(), self.num_cls))

        return id_to_samples


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = StanfordProducts(data_root, transform=img_transform)
    num_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
