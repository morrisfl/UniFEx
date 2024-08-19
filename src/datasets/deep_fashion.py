import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class DeepFashion(Dataset):
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
        self.root = os.path.join(root, "deepfashion")
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.num_cls = num_cls

        self.id_to_samples = self._sampling()

        self.class_ids = sorted(list(self.id_to_samples.keys()))
        self.num_classes = len(self.class_ids)
        self.id_to_label = {cat_id: idx + offset for idx, cat_id in enumerate(self.id_to_samples.keys())}

        self.samples = []
        for samples in self.id_to_samples.values():
            self.samples.extend(samples)

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        class_id = img_path.split("/")[-2]
        label = self.id_to_label[class_id]

        if self.augmentation:
            img = self.augmentation(img)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def _sampling(self):
        with open(os.path.join(self.root, "deepfashion_train.json")) as f:
            annotations = json.load(f)

        id_to_samples = {}
        for group_key, group_dict in annotations.items():
            for id, samples in group_dict.items():
                id_to_samples[id] = samples

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
            remaining_ids = list(id_to_samples.keys())
            while self.num_cls < len(remaining_ids):
                chosen_id = random.choice(remaining_ids)
                id_to_samples.pop(chosen_id)
                remaining_ids = list(id_to_samples.keys())

        return id_to_samples


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = DeepFashion(data_root, transform=img_transform,)
    number_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
