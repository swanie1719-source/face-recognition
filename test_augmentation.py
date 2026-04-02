# test_augmentation.py
import cv2
import numpy as np
import sys
sys.path.insert(0, '.')

from data.augmentation import FaceAugmentor, TrainTransform
from PIL import Image

# 造假图测试
fake_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
aug = FaceAugmentor(face_size=112)

# 测试各个增强方法
flipped    = aug.horizontal_flip(fake_img)
rotated    = aug.random_rotate(fake_img)
jittered   = aug.color_jitter(fake_img)
cutout_img = aug.cutout(fake_img)
masked     = aug.mask_lower_face(fake_img)
small_face = aug.small_face_scale(fake_img)
full_aug   = aug.augment(fake_img)

print(f"原始尺寸:     {fake_img.shape}")
print(f"翻转后:       {flipped.shape}")
print(f"旋转后:       {rotated.shape}")
print(f"色彩抖动后:   {jittered.shape}")
print(f"Cutout后:     {cutout_img.shape}")
print(f"口罩遮挡后:   {masked.shape}")
print(f"小尺度人脸后: {small_face.shape}")
print(f"完整增强后:   {full_aug.shape}")

# 测试 TrainTransform
transform = TrainTransform(face_size=112)
pil_img   = Image.fromarray(
    np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
)
tensor = transform(pil_img)
print(f"\nTrainTransform 输出: {tensor.shape}")
print(f"值域: [{tensor.min():.2f}, {tensor.max():.2f}]")
print("M3 数据增强模块验证通过!")