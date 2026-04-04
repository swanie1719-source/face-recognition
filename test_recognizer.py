# test_recognizer.py
import torch
import sys
sys.path.insert(0, '.')
from models.recognizer import FaceRecognizer, ArcFaceLoss, FaceModel

# 测试特征提取器
recognizer = FaceRecognizer(embedding_dim=512, pretrained=False)
fake_input = torch.randn(4, 3, 112, 112)
features   = recognizer(fake_input)
print(f"输入shape:    {fake_input.shape}")
print(f"特征shape:    {features.shape}")       # [4, 512]
print(f"特征L2范数:   {features.norm(dim=1)}")  # 应该全是1.0

# 测试 ArcFace 损失
arcface = ArcFaceLoss(in_features=512, num_classes=100)
labels  = torch.randint(0, 100, (4,))
loss    = arcface(features, labels)
print(f"\nArcFace loss: {loss.item():.4f}")

# 测试完整模型
model  = FaceModel(num_classes=100, pretrained=False)
feats, loss = model(fake_input, labels)
print(f"\nFaceModel 输出特征: {feats.shape}")
print(f"FaceModel loss:     {loss.item():.4f}")
print("\nM5 模型验证通过!")