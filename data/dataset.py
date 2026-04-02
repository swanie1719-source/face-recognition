import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path


class FaceDataset(Dataset):
    """
    通用人脸数据集加载器
    支持标准(LFW/MS1MV3)和非标准(IJB-C)数据集
    目录结构要求:
        root/
            person_001/
                img1.jpg
                img2.jpg
            person_002/
                ...
    """

    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode

        self.samples = []   # [(图片路径, 标签), ...]
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}

        self._scan_dataset()

    def _scan_dataset(self):
        """扫描目录，建立 (路径, 标签) 列表"""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

        # 找所有子目录（每个子目录 = 一个人）
        persons = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        if not persons:
            raise ValueError(f"数据集目录为空或格式不对: {self.root_dir}")

        print(f"发现 {len(persons)} 个人物类别")

        for idx, person_dir in enumerate(persons):
            self.class_names.append(person_dir.name)
            self.class_to_idx[person_dir.name] = idx

            img_files = [
                f for f in person_dir.iterdir()
                if f.suffix.lower() in valid_exts
            ]

            for img_path in img_files:
                self.samples.append((str(img_path), idx))
                self.labels.append(idx)

        print(f"共加载 {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图片，转 RGB
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_count(self):
        return len(self.class_names)


class FaceVerificationDataset(Dataset):
    """
    用于验证阶段的数据集（LFW格式）
    加载图片对，判断是否同一个人
    LFW pairs.txt 格式:
        同一人: name  img1_num  img2_num
        不同人: name1  img_num  name2  img_num
    """

    def __init__(self, root_dir, pairs_file, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.pairs = []
        self.labels = []  # 1=同一人, 0=不同人

        self._load_pairs(pairs_file)

    def _load_pairs(self, pairs_file):
        with open(pairs_file, 'r') as f:
            lines = f.readlines()

        # 跳过第一行（配置行）
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                # 同一个人
                name, n1, n2 = parts
                p1 = self._get_img_path(name, int(n1))
                p2 = self._get_img_path(name, int(n2))
                self.pairs.append((p1, p2))
                self.labels.append(1)
            elif len(parts) == 4:
                # 不同的人
                name1, n1, name2, n2 = parts
                p1 = self._get_img_path(name1, int(n1))
                p2 = self._get_img_path(name2, int(n2))
                self.pairs.append((p1, p2))
                self.labels.append(0)

        print(f"加载 {len(self.pairs)} 对验证图片")
        same = sum(self.labels)
        print(f"  同一人: {same}, 不同人: {len(self.labels) - same}")

    def _get_img_path(self, name, num):
        # LFW 文件名格式: Name_0001.jpg
        filename = f"{name}_{num:04d}.jpg"
        return str(self.root_dir / name / filename)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        label = self.labels[idx]

        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


def build_dataloader(dataset, batch_size=64,
                     shuffle=True, num_workers=2):
    """统一的 DataLoader 构建函数"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,        # 加速 GPU 数据传输
        drop_last=True if shuffle else False
    )