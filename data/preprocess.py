import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class FacePreprocessor:
    """
    图像预处理流水线
    负责把原始人脸图片处理成模型可以接受的格式
    处理步骤：resize → CLAHE增强 → 高斯去噪 → 归一化
    """

    def __init__(self, face_size=112):
        self.face_size = face_size

        # 归一化参数（和训练时保持一致）
        self.mean = [0.5, 0.5, 0.5]
        self.std  = [0.5, 0.5, 0.5]

        # CLAHE 对比度增强器
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    # ── 步骤1：尺寸归一化 ──────────────────────────────
    def resize(self, img):
        """
        统一缩放到 face_size x face_size
        img: numpy array BGR 或 RGB
        """
        if img.shape[0] == self.face_size and \
           img.shape[1] == self.face_size:
            return img
        return cv2.resize(
            img,
            (self.face_size, self.face_size),
            interpolation=cv2.INTER_LINEAR
        )

    # ── 步骤2：CLAHE 对比度增强 ────────────────────────
    def clahe_enhance(self, img):
        """
        CLAHE（限制对比度自适应直方图均衡化）
        解决光照不均问题，对低光照/强光照图片效果明显
        只对亮度通道操作，不改变色彩
        img: numpy array BGR
        """
        # BGR → LAB色彩空间（L=亮度，A/B=色彩）
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 只对亮度通道做均衡化
        l_enhanced = self.clahe.apply(l)

        # 合并回去
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return result

    # ── 步骤3：高斯滤波去噪 ────────────────────────────
    def denoise(self, img):
        """
        轻度高斯模糊去除图像噪声
        kernel_size=3 保留细节的同时去除椒盐噪声
        img: numpy array BGR
        """
        return cv2.GaussianBlur(img, (3, 3), 0)

    # ── 步骤4：归一化到[-1, 1] ─────────────────────────
    def normalize(self, img):
        """
        像素值从[0,255] → [-1,1]
        公式：(pixel/255 - 0.5) / 0.5
        img: numpy array BGR
        返回：torch tensor [3, H, W]
        """
        # BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToTensor(),           # [0,255] → [0,1]
            transforms.Normalize(
                mean=self.mean,
                std=self.std
            )                                # [0,1] → [-1,1]
        ])

        # numpy → PIL → tensor
        pil_img = Image.fromarray(img_rgb)
        tensor = transform(pil_img)          # [3, 112, 112]
        return tensor

    # ── 完整流水线（一步到位）──────────────────────────
    def process(self, img):
        """
        完整预处理流水线
        输入：numpy array BGR，任意尺寸
        输出：torch tensor [3, 112, 112]，值域[-1,1]
        """
        img = self.resize(img)          # 步骤1
        img = self.clahe_enhance(img)   # 步骤2
        img = self.denoise(img)         # 步骤3
        tensor = self.normalize(img)    # 步骤4
        return tensor

    def process_pil(self, pil_img):
        """
        接受 PIL Image 输入的版本
        用于 torchvision Dataset 的 transform
        """
        img = cv2.cvtColor(
            np.array(pil_img),
            cv2.COLOR_RGB2BGR
        )
        return self.process(img)

    # ── 批量处理 ───────────────────────────────────────
    def process_batch(self, imgs):
        """
        批量处理多张图片
        imgs: list of numpy array BGR
        返回：torch tensor [N, 3, 112, 112]
        """
        tensors = [self.process(img) for img in imgs]
        return torch.stack(tensors)     # [N, 3, 112, 112]


def get_train_transform(face_size=112):
    """
    训练阶段的 transform（给 Dataset 用）
    包含预处理，数据增强在 augmentation.py 里另外加
    """
    preprocessor = FacePreprocessor(face_size=face_size)
    return preprocessor.process_pil


def get_val_transform(face_size=112):
    """
    验证/测试阶段的 transform
    只做预处理，不做数据增强
    """
    preprocessor = FacePreprocessor(face_size=face_size)
    return preprocessor.process_pil