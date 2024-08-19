import torch.nn as nn
from torchvision import transforms

from model.backbone import get_foundational_model
from model.head import build_head
from model.neck import Neck


class TrainModel(nn.Module):
    def __init__(self, config, classes, cls_dist=None):
        super(TrainModel, self).__init__()
        self.backbone = get_foundational_model(config)
        try:
            self.mean = self.backbone.mean
            self.std = self.backbone.std
            self.output_dim = self.backbone.out_dim
        except AttributeError:
            self.mean = config.TRANSFORM.mean
            self.std = config.TRANSFORM.std
            self.output_dim = config.MODEL.BACKBONE.output_dim

        self.neck = Neck(config, self.output_dim)
        self.head = build_head(config, classes, cls_dist)

    def forward(self, img, labels):
        img_norm = self.normalize(img)
        feat = self.backbone(img_norm)
        embed = self.neck(feat)
        logits = self.head(embed, labels)
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def normalize(self, x):
        return transforms.functional.normalize(x, self.mean, self.std)


class InferenceModel(nn.Module):
    def __init__(self, config):
        super(InferenceModel, self).__init__()
        self.backbone = get_foundational_model(config)
        try:
            self.mean = self.backbone.mean
            self.std = self.backbone.std
            self.img_size = self.backbone.img_size
            self.output_dim = self.backbone.out_dim
        except AttributeError:
            self.mean = config.TRANSFORM.mean
            self.std = config.TRANSFORM.std
            self.img_size = config.TRANSFORM.size
            self.output_dim = config.MODEL.BACKBONE.output_dim

        self.head = Neck(config, self.output_dim)

    def normalize(self, x):
        x = x / 255.0
        x = transforms.functional.normalize(x, self.mean, self.std)
        return x

    def resize(self, x):
        x = transforms.functional.resize(x, [self.img_size, self.img_size],
                                         interpolation=transforms.InterpolationMode.BICUBIC,
                                         antialias=True)
        return x

    def forward(self, img):
        img = self.resize(img)
        img = self.normalize(img)
        features = self.backbone(img)
        embedding = self.head(features)
        return embedding


def get_inference_model(config, model):
    model.eval()
    try:
        backbone_weights = model.backbone.state_dict()
        neck_weights = model.neck.state_dict()
    except AttributeError:
        backbone_weights = model.module.backbone.state_dict()
        neck_weights = model.module.neck.state_dict()

    inference_model = InferenceModel(config)
    inference_model.backbone.load_state_dict(backbone_weights)
    inference_model.head.load_state_dict(neck_weights)

    return inference_model
