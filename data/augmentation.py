import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import random


class FaceAugmentor:
    """
    数据增强流水线
    训练阶段使用，验证/测试阶段不使用
    """

    def __init__(self, face_size=112):
        self.face_size = face_size

    # ── 1. 水平翻转 ────────────────────────────────────
    def horizontal_flip(self, img, p=0.5):
        """随机水平翻转，模拟左右脸"""
        if random.random() < p:
            return cv2.flip(img, 1)
        return img

    # ── 2. 随机旋转 ────────────────────────────────────
    def random_rotate(self, img, degrees=15, p=0.5):
        """
        随机旋转 ±degrees 度
        模拟头部轻微倾斜
        """
        if random.random() > p:
            return img
        angle = random.uniform(-degrees, degrees)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

    # ── 3. 色彩抖动 ────────────────────────────────────
    def color_jitter(self, img, brightness=0.3,
                     contrast=0.3, p=0.5):
        """
        随机调整亮度和对比度
        模拟不同光照环境
        """
        if random.random() > p:
            return img

        # 亮度
        b_factor = 1 + random.uniform(-brightness, brightness)
        # 对比度
        c_factor = 1 + random.uniform(-contrast, contrast)

        img = img.astype(np.float32)
        img = img * b_factor                    # 亮度
        img = (img - 128) * c_factor + 128      # 对比度
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    # ── 4. 随机遮挡 Cutout ─────────────────────────────
    def cutout(self, img, p=0.3,
               min_size=0.1, max_size=0.3):
        """
        随机遮挡图片的一个矩形区域
        模拟局部遮挡（帽子、手、头发等）
        """
        if random.random() > p:
            return img

        h, w = img.shape[:2]
        cut_h = int(h * random.uniform(min_size, max_size))
        cut_w = int(w * random.uniform(min_size, max_size))

        # 随机位置
        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)

        img = img.copy()
        img[y:y+cut_h, x:x+cut_w] = 128    # 填充灰色
        return img

    # ── 5. 口罩遮挡（下半脸）★ ───────────────────────
    def mask_lower_face(self, img, p=0.3):
        """
        遮挡下半脸，模拟口罩效果
        专门增强口罩场景识别能力
        """
        if random.random() > p:
            return img

        h, w = img.shape[:2]
        # 遮挡下半部分（从40%高度开始到底部）
        start_y = int(h * 0.4)
        img = img.copy()
        img[start_y:, :] = 128
        return img

    # ── 6. 小尺度人脸生成 ★★（核心加分项）────────────
    def small_face_scale(self, img, p=0.3,
                          min_scale=0.05, max_scale=0.3):
        """
        缩放生成小尺度人脸样本
        原理：把人脸缩小后再放大回原尺寸
             模拟远距离/低分辨率人脸
        min_scale: 最小缩放比例（0.05=缩到5%大小）
        max_scale: 最大缩放比例（0.3=缩到30%大小）
        """
        if random.random() > p:
            return img

        h, w = img.shape[:2]
        scale = random.uniform(min_scale, max_scale)

        # 缩小
        small_h = max(int(h * scale), 4)
        small_w = max(int(w * scale), 4)
        small = cv2.resize(
            img, (small_w, small_h),
            interpolation=cv2.INTER_AREA      # 缩小用AREA，质量更好
        )

        # 放大回原尺寸（有意保留模糊，模拟低分辨率）
        restored = cv2.resize(
            small, (w, h),
            interpolation=cv2.INTER_LINEAR    # 放大用LINEAR，产生模糊
        )
        return restored

    # ── 完整增强流水线 ─────────────────────────────────
    def augment(self, img):
        """
        训练阶段完整增强流程
        输入：numpy array BGR
        输出：numpy array BGR（增强后）
        """
        img = self.horizontal_flip(img, p=0.5)
        img = self.random_rotate(img, degrees=15, p=0.5)
        img = self.color_jitter(img, p=0.5)
        img = self.cutout(img, p=0.3)
        img = self.mask_lower_face(img, p=0.2)
        img = self.small_face_scale(img, p=0.3)
        return img


class TrainTransform:
    """
    训练阶段完整 transform
    增强 + 预处理，给 Dataset 的 transform 参数用
    """

    def __init__(self, face_size=112):
        from data.preprocess import FacePreprocessor
        self.augmentor   = FaceAugmentor(face_size)
        self.preprocessor = FacePreprocessor(face_size)

    def __call__(self, pil_img):
        # PIL → numpy BGR
        img = cv2.cvtColor(
            np.array(pil_img),
            cv2.COLOR_RGB2BGR
        )
        # 增强
        img = self.augmentor.augment(img)
        # 预处理 → tensor
        return self.preprocessor.process(img)


class ValTransform:
    """
    验证/测试阶段 transform
    只做预处理，不做增强
    """

    def __init__(self, face_size=112):
        from data.preprocess import FacePreprocessor
        self.preprocessor = FacePreprocessor(face_size)

    def __call__(self, pil_img):
        return self.preprocessor.process_pil(pil_img)