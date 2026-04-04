import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math


# ── ArcFace 损失函数 ───────────────────────────────────
class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    论文：ArcFace: Additive Angular Margin Loss for Deep Face Recognition

    原理：在特征向量和权重矩阵之间的夹角上加一个margin
         强迫同一个人的特征向量更加聚集
         不同人的特征向量更加分离

    参数：
      in_features : 输入特征维度（512）
      num_classes : 训练集人物数量
      s           : 特征缩放因子（默认64）
      m           : 角度边距（默认0.5，约28.6度）
    """

    def __init__(self, in_features=512, num_classes=2000,
                 s=64.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m

        # 可学习的权重矩阵
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        # 预计算角度边距的cos和sin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        """
        features : [B, 512] L2归一化后的特征向量
        labels   : [B] 人物标签
        """
        # L2 归一化特征和权重
        features_norm = F.normalize(features, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)

        # 计算余弦相似度 cos(θ)
        cosine = F.linear(features_norm, weight_norm)  # [B, num_classes]
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # 计算 sin(θ)
        sine = torch.sqrt(1.0 - cosine ** 2)

        # 计算 cos(θ + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 当 θ + m > π 时，用近似值避免梯度消失
        phi = torch.where(cosine > self.th, phi,
                          cosine - self.mm)

        # one-hot 编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # 对目标类别加 margin，非目标类别保持原值
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.s

        # 交叉熵损失
        loss = F.cross_entropy(output, labels)
        return loss


# ── ResNet50 特征提取器 ────────────────────────────────
class FaceRecognizer(nn.Module):
    """
    ResNet50 人脸特征提取器
    输入：[B, 3, 112, 112] 对齐人脸图片
    输出：[B, 512] L2归一化特征向量
    """

    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()

        # 加载预训练 ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        # 去掉最后的分类层，保留特征提取部分
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-2]
        )  # 输出 [B, 2048, 4, 4]

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, 2048, 1, 1]

        # BN → FC → BN
        self.bn1 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # 初始化
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x: [B, 3, 112, 112]
        返回: [B, 512] L2归一化特征向量
        """
        x = self.backbone(x)  # [B, 2048, 4, 4]
        x = self.bn1(x)
        x = self.gap(x)  # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 2048]
        x = self.fc(x)  # [B, 512]
        x = self.bn2(x)

        # L2 归一化（使得后续余弦相似度计算有意义）
        x = F.normalize(x, dim=1)
        return x


# ── 完整训练模型（特征提取器 + ArcFace）──────────────
class FaceModel(nn.Module):
    """
    训练阶段使用，包含特征提取器和ArcFace损失
    推理阶段只用 FaceRecognizer
    """

    def __init__(self, num_classes=2000,
                 embedding_dim=512, pretrained=True):
        super().__init__()
        self.recognizer = FaceRecognizer(
            embedding_dim=embedding_dim,
            pretrained=pretrained
        )
        self.arcface = ArcFaceLoss(
            in_features=embedding_dim,
            num_classes=num_classes
        )

    def forward(self, x, labels=None):
        features = self.recognizer(x)
        if labels is not None:
            loss = self.arcface(features, labels)
            return features, loss
        return features