import json
import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class METArt(Dataset):
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
        self.root = os.path.join(root, "met_dataset")
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.num_cls = num_cls
        self.data_amount = data_amount

        self.samples, self.class_ids = self._sampling()
        self.num_classes = len(self.class_ids)
        self.ids_to_label = {class_id: idx + offset for idx, class_id in enumerate(self.class_ids)}

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.root, "MET", item["path"])

        img = Image.open(img_path).convert("RGB")
        label = self.ids_to_label[item["id"]]

        if self.augmentation:
            img = self.augmentation(img)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def _sampling(self):
        annotation_path = os.path.join(self.root, "ground_truth", "MET_database.json")

        with open(annotation_path) as f:
            samples = json.load(f)

        if self.data_amount:
            samples = random.sample(samples, self.data_amount)

        if self.min_samples or self.max_samples:
            samples_per_class = defaultdict(int)
            for item in samples:
                samples_per_class[item["id"]] += 1
            discard_classes = []
            if self.min_samples:
                discard_classes += [key for key, value in samples_per_class.items() if value < self.min_samples]
            if self.max_samples:
                discard_classes += [key for key, value in samples_per_class.items() if value > self.max_samples]

            samples = [item for item in samples if item["id"] not in discard_classes]

        class_ids = []
        for item in samples:
            if item["id"] not in class_ids:
                class_ids.append(item["id"])

        if self.num_cls is not None:
            if len(class_ids) >= self.num_cls:
                class_ids = random.sample(class_ids, self.num_cls)
                samples = [item for item in samples if item["id"] in class_ids]

        return samples, sorted(class_ids)


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = METArt(data_root, transform=img_transform, data_amount=10000)
    number_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
