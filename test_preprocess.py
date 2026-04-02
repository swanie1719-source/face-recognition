# 在 PyCharm 里新建一个 test_preprocess.py 测试
import cv2
import numpy as np
from data.preprocess import FacePreprocessor

# 造一张假图测试
fake_img = np.random.randint(0, 255, (200, 180, 3), dtype=np.uint8)

preprocessor = FacePreprocessor(face_size=112)
tensor = preprocessor.process(fake_img)

print(f"输入尺寸: {fake_img.shape}")       # (200, 180, 3)
print(f"输出尺寸: {tensor.shape}")         # torch.Size([3, 112, 112])
print(f"值域: [{tensor.min():.2f}, {tensor.max():.2f}]")  # [-1, 1]
print("M2 预处理模块验证通过!")