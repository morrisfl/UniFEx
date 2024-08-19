import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, config, num_classes, class_dist=None, dynamic_m=False,):
        """
        Implementation of:
            ArcFace (https://arxiv.org/abs/1801.07698),
            Sub-centered ArcFace (https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf), and
            ArcFace with Dynamic Margin based on class distribution (own approach).

        Args:
            config (yacs.config.CfgNode): configuration file with specifications for the head
            num_classes (int): size of each output sample
            class_dist (dict): dictionary of class distribution (used to calculate dynamic margins)
            dynamic_m (bool): whether to use dynamic margins

        Return:
            torch.Tensor: logits
        """
        super().__init__()
        self.in_features = config.MODEL.embedding_dim
        self.out_features = num_classes

        self.dynamic_m = dynamic_m
        if self.dynamic_m:
            if class_dist is None:
                raise ValueError("class_dist must be provided when using dynamic_m")
            else:
                self.cls_dist = torch.Tensor(list(class_dist.values()))
                self.m = self._calc_dyn_margins(config.MODEL.HEAD.m_max, config.MODEL.HEAD.m_min)
                self.th = torch.cos(math.pi - self.m)
                self.mm = torch.sin(math.pi - self.m) * self.m
        else:
            self.m = config.MODEL.HEAD.m
            self.th = math.cos(math.pi - self.m)
            self.mm = math.sin(math.pi - self.m) * self.m

        self.s = config.MODEL.HEAD.s

        self.k = config.MODEL.HEAD.k
        self.weight = nn.Parameter(torch.Tensor(num_classes * self.k, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        if self.k > 1:
            cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
            cosine_all = cosine_all.view(-1, self.out_features, self.k)
            cos_theta, _ = torch.max(cosine_all, dim=self.k - 1)
            cos_theta = cos_theta.clamp(-1, 1)
        else:
            cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
            cos_theta = cos_theta.clamp(-1, 1)

        if self.dynamic_m:
            m = self.m[labels][:, None].to(features.device)
            th = self.th[labels][:, None].to(features.device)
            mm = self.mm[labels][:, None].to(features.device)
        else:
            m = self.m
            th = self.th
            mm = self.mm

        theta = torch.acos(cos_theta)
        theta_m = theta + m
        cos_theta_m = torch.cos(theta_m)

        final_logits = torch.where(cos_theta > th, cos_theta_m, cos_theta - mm)

        one_hot = F.one_hot(labels.long(), num_classes=self.out_features).float()
        output = (one_hot * final_logits) + ((1.0 - one_hot) * cos_theta)

        return self.s * output

    def _calc_dyn_margins(self, m_max, m_min):
        cls_dist = (self.cls_dist - torch.min(self.cls_dist)) / (torch.max(self.cls_dist) - torch.min(self.cls_dist))
        return m_min + 0.5 * (m_max - m_min) * (1 + torch.cos(math.pi * cls_dist))


class LiArcFaceHead(nn.Module):
    def __init__(self, config, num_classes,):
        """
        Implementation of Li-ArcFace (https://arxiv.org/abs/1907.12256).

        Args:
            config (yacs.config.CfgNode): configuration file with specifications for the head
            num_classes (int): size of each output sample

        Return:
            torch.Tensor: logits
        """
        super().__init__()
        self.in_features = config.MODEL.embedding_dim
        self.out_features = num_classes

        self.m = config.MODEL.HEAD.m
        self.s = config.MODEL.HEAD.s

        self.weight = nn.Parameter(torch.Tensor(num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)

        theta = torch.acos(cos_theta)
        theta_m = theta + self.m

        one_hot = F.one_hot(labels.long(), num_classes=self.out_features).float()
        output = (one_hot * theta_m) + ((1.0 - one_hot) * theta)

        return self.s * (math.pi - 2 * output) / math.pi


class AdaCosHead(nn.Module):
    def __init__(self, config, num_classes,):
        """
        Implementation of AdaCos (https://arxiv.org/abs/1905.00292).

        Args:
            config (yacs.config.CfgNode): configuration file with specifications for the head
            num_classes (int): size of each output sample.

        Return:
            torch.Tensor: logits
        """
        super().__init__()
        self.in_features = config.MODEL.embedding_dim
        self.out_features = num_classes

        self.dynamic_s = config.MODEL.HEAD.dynamic_s
        self.s = math.sqrt(2) * math.log(num_classes - 1)

        self.weight = nn.Parameter(torch.Tensor(num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        theta = torch.acos(cos_theta)
        one_hot = F.one_hot(labels.long(), num_classes=self.out_features).float()

        if self.dynamic_s:
            with torch.no_grad():
                b_avg = torch.where(one_hot < 1, torch.exp(self.s * cos_theta), torch.zeros_like(cos_theta))
                b_avg = torch.sum(b_avg) / cos_theta.size(0)
                theta_med = torch.median(theta[one_hot == 1])
                self.s = torch.log(b_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))

        return self.s * cos_theta


class CurricularFaceHead(nn.Module):
    def __init__(self, config, num_classes,):
        """
        Implementation of CurricularFace (https://arxiv.org/abs/2004.00288).

        Args:
            config (yacs.config.CfgNode): configuration file with specifications for the head
            num_classes (int): size of each output sample.

        Return:
            torch.Tensor: logits
        """
        super().__init__()
        self.in_features = config.MODEL.embedding_dim
        self.out_features = num_classes

        self.m = config.MODEL.HEAD.m
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        self.s = config.MODEL.HEAD.s

        self.register_buffer("t", torch.zeros(1))

        self.weight = nn.Parameter(torch.Tensor(num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)

        target_cos_theta = cos_theta[torch.arange(0, features.size(0)), labels].view(-1, 1)
        target_theta = torch.acos(target_cos_theta)
        target_cos_theta_m = torch.cos(target_theta + self.m)
        mask = cos_theta > target_cos_theta_m
        final_target_cos_theta_m = torch.where(target_cos_theta > self.th, target_cos_theta_m,
                                               target_cos_theta - self.mm)

        with torch.no_grad():
            self.t = target_cos_theta.mean() * 0.01 + (1 - 0.01) * self.t

        hard_example = cos_theta[mask]
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_cos_theta_m)

        return self.s * cos_theta


class AdaFaceHead(nn.Module):
    def __init__(self, config, num_classes,):
        """
        Implementation of AdaFace (https://arxiv.org/abs/2204.00964).

        Args:
            config (yacs.config.CfgNode): configuration file with specifications for the head
            num_classes (int): size of each output sample.

        Return:
            torch.Tensor: logits
        """
        super().__init__()
        self.in_features = config.MODEL.embedding_dim
        self.out_features = num_classes

        self.s = config.MODEL.HEAD.s
        self.m = config.MODEL.HEAD.m
        self.h = config.MODEL.HEAD.h

        # ema prep
        self.t_alpha = config.MODEL.HEAD.t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)

        self.weight = nn.Parameter(torch.Tensor(num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.eps = 1e-3

    def forward(self, features, labels):
        norms = torch.norm(features, dim=1)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1 + self.eps, 1 - self.eps)

        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()

        # update batch_mean batch_std
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # g_angular
        g_angular = self.m * margin_scaler * - 1
        ma = torch.zeros(labels.size()[0], cos_theta.size()[1], device=cos_theta.device)
        ma.scatter_(1, labels.reshape(-1, 1), 1.0)
        ma = ma * g_angular[:, None]
        theta = torch.acos(cos_theta)
        theta_ma = torch.clip(theta + ma, min=self.eps, max=math.pi - self.eps)
        cos_theta_ma = torch.cos(theta_ma)

        # g_additive
        g_add = self.m + (self.m * margin_scaler)
        mc = torch.zeros(labels.size()[0], cos_theta.size()[1], device=cos_theta.device)
        mc.scatter_(1, labels.reshape(-1, 1), 1.0)
        mc = mc * g_add[:, None]

        final_logits = cos_theta_ma - mc

        return self.s * final_logits


def build_head(config, num_classes, cls_dist):
    if config.MODEL.HEAD.name == "ArcFace":
        return ArcFaceHead(config, num_classes, dynamic_m=False)
    elif config.MODEL.HEAD.name == "DynM-ArcFace":
        return ArcFaceHead(config, num_classes, class_dist=cls_dist, dynamic_m=True)
    elif config.MODEL.HEAD.name == "LiArcFace":
        return LiArcFaceHead(config, num_classes)
    elif config.MODEL.HEAD.name == "AdaCos":
        return AdaCosHead(config, num_classes)
    elif config.MODEL.HEAD.name == "CurricularFace":
        return CurricularFaceHead(config, num_classes)
    elif config.MODEL.HEAD.name == "AdaFace":
        return AdaFaceHead(config, num_classes)
    else:
        raise NotImplementedError(f"Unsupported head: {config.MODEL.HEAD.name}")


if __name__ == "__main__":
    pass

