import torch.nn as nn


class Neck(nn.Module):
    def __init__(self, config, in_features=None):
        super().__init__()
        if not None:
            self.in_features = in_features
        else:
            self.in_features = config.MODEL.BACKBONE.output_dim
        self.out_features = config.MODEL.embedding_dim
        self.neck_type = config.MODEL.NECK.neck_type
        self.dropout = config.MODEL.NECK.dropout

        if self.neck_type == "proj_layer":
            self.neck = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.in_features, self.out_features)
            )
        elif self.neck_type == "pooling":
            self.neck = nn.AdaptiveAvgPool1d(self.out_features)
        else:
            raise NotImplementedError(f"Unsupported neck type: {self.neck_type}")

    def forward(self, x):
        return self.neck(x)
