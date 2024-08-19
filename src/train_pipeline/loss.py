import torch.nn as nn


def build_loss(config):
    if config.LOSS.name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Unsupported loss: {config.LOSS.name}")


if __name__ == "__main__":
    pass
