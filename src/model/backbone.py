import open_clip
import torch
from torch import nn
import timm


class OpenClipViT(nn.Module):
    def __init__(self, model_name, pretrained, with_proj_layer):
        super().__init__()
        model, train_transform, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.vit = model.visual

        if not with_proj_layer:
            self.vit.proj = None
            self.out_dim = self.vit.transformer.width
        else:
            self.out_dim = self.vit.output_dim

        self.img_size = self.vit.image_size[-1]
        self.mean = train_transform.transforms[-1].mean
        self.std = train_transform.transforms[-1].std

    def forward(self, x):
        embedding = self.vit(x)

        return embedding


class EVA02ViT(nn.Module):
    def __init__(self, model_name, with_proj_layer):
        super().__init__()
        if with_proj_layer:
            self.vit = timm.create_model(model_name, pretrained=True)
            self.out_dim = self.vit.num_classes
        else:
            self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.out_dim = self.vit.embed_dim

        self.img_size = self.vit.default_cfg["input_size"][-1]
        self.mean = self.vit.default_cfg["mean"]
        self.std = self.vit.default_cfg["std"]

    def forward(self, x):
        embedding = self.vit(x)

        return embedding


class DINOv2ViT(nn.Module):
    def __init__(self, model_name, img_size=None):
        super().__init__()
        if img_size is None:
            self.vit = timm.create_model(model_name, pretrained=True)
            self.img_size = self.vit.default_cfg["input_size"][-1]
        else:
            self.vit = timm.create_model(model_name, pretrained=True, img_size=img_size)
            self.img_size = img_size

        self.out_dim = self.vit.embed_dim
        self.mean = self.vit.default_cfg["mean"]
        self.std = self.vit.default_cfg["std"]

    def forward(self, x):
        embedding = self.vit(x)

        return embedding


class SAMViT(nn.Module):
    def __init__(self, model_name, image_size=224):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        self.vit.neck = torch.nn.Identity()
        self.out_dim = self.vit.embed_dim
        self.img_size = image_size
        self.mean = self.vit.default_cfg["mean"]
        self.std = self.vit.default_cfg["std"]

    def forward(self, x):
        embeddings = self.vit(x)

        return embeddings


class OpenClipConvNext(nn.Module):
    def __init__(self, model_name, pretrained, with_proj_layer):
        super().__init__()
        model, train_transform, _ = open_clip.create_model_and_transforms(model_name, pretrained)

        if with_proj_layer:
            self.encoder = model.visual
            try:
                self.out_dim = self.encoder.head.proj.out_features
            except AttributeError:
                self.encoder.head.mlp.drop2 = nn.Dropout(p=0.0)
                self.out_dim = self.encoder.head.mlp.fc2.out_features
        else:
            self.encoder = model.visual.trunk
            self.out_dim = self.encoder.num_features

        self.img_size = model.visual.image_size[0]
        self.mean = model.visual.image_mean
        self.std = model.visual.image_std

    def forward(self, x):
        embedding = self.encoder(x)

        return embedding


class SigLIPViT(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        self.img_size = self.vit.default_cfg["input_size"][-1]
        self.out_dim = self.vit.num_features
        self.mean = self.vit.default_cfg["mean"]
        self.std = self.vit.default_cfg["std"]

    def forward(self, x):
        embedding = self.vit(x)

        return embedding


def get_foundational_model(config):
    if config.MODEL.BACKBONE.type in ["clip", "clipav2", "meta-clip"]:
        model = OpenClipViT(config.MODEL.BACKBONE.model_name, config.MODEL.BACKBONE.weights,
                            config.MODEL.BACKBONE.proj_layer)

    elif config.MODEL.BACKBONE.type == "eva02":
        model = EVA02ViT(config.MODEL.BACKBONE.model_name, config.MODEL.BACKBONE.proj_layer)

    elif config.MODEL.BACKBONE.type == "dinov2":
        model = DINOv2ViT(config.MODEL.BACKBONE.model_name, config.TRANSFORM.size)

    elif config.MODEL.BACKBONE.type == "sam":
        model = SAMViT(config.MODEL.BACKBONE.model_name, config.TRANSFORM.size)

    elif config.MODEL.BACKBONE.type == "clip_convnext":
        model = OpenClipConvNext(config.MODEL.BACKBONE.model_name, config.MODEL.BACKBONE.weights,
                                 config.MODEL.BACKBONE.proj_layer)

    elif config.MODEL.BACKBONE.type == "siglip":
        model = SigLIPViT(config.MODEL.BACKBONE.model_name)

    else:
        raise NotImplementedError(f"Unsupported foundational model: {config.MODEL.BACKBONE.type}")

    return model


if __name__ == "__main__":
    pass
