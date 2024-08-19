from torchvision.transforms import transforms


def build_transforms(config):
    size = config.TRANSFORM.size
    if config.TRANSFORM.name == "openai-clip":
        train_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])

    elif config.TRANSFORM.name in ["openclip", "eva-clip"]:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), ratio=(1.0, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
            ])

    elif config.TRANSFORM.name == "siglip":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4]),
            transforms.ToTensor()
        ])
    elif config.TRANSFORM.name == "custom":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), ratio=(1.0, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4]),
            transforms.ToTensor()
        ])
    else:
        raise NotImplementedError(f"Unsupported transformation: {config.TRANSFORM.name}")

    return train_transform


if __name__ == "__main__":
    pass
