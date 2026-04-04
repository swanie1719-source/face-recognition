import os
import struct
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


class MS1MStreamDataset(Dataset):
    """
    流式读取 MS1M .rec 文件
    训练时按需解码，不预先解压
    不占额外磁盘空间
    label 位置：offset=12，float32 转 int
    """

    def __init__(self, rec_path, idx_path,
                 num_identities=2000, transform=None, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        self.rec_path  = rec_path
        self.transform = transform

        print("读取idx索引...")
        self.offsets = []
        with open(idx_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.offsets.append(int(parts[1]))
        print(f"总记录数: {len(self.offsets)}")

        self._build_index(num_identities, seed)
        print(f"训练集: {len(self.valid_indices)} 张, "
              f"{self.num_classes} 个identity")

    def _build_index(self, num_identities, seed):
        random.seed(seed)

        print("扫描label...")
        all_labels = []
        with open(self.rec_path, 'rb') as f:
            for offset in tqdm(self.offsets, desc="扫描"):
                f.seek(offset)
                data = f.read(20)
                if len(data) < 20:
                    all_labels.append(-1)
                    continue
                magic = struct.unpack('<I', data[0:4])[0]
                if magic != 0xced7230a:
                    all_labels.append(-1)
                    continue
                # label在offset=12，float32转int（已验证）
                val = struct.unpack('<f', data[12:16])[0]
                all_labels.append(int(val))

        valid_ids = list(set(l for l in all_labels if l >= 0))
        print(f"发现 {len(valid_ids)} 个唯一identity")
        print(f"label范围: {min(valid_ids)} ~ {max(valid_ids)}")

        selected = set(random.sample(
            valid_ids, min(num_identities, len(valid_ids))
        ))
        self.id_to_new = {
            old: new for new, old
            in enumerate(sorted(selected))
        }
        self.num_classes   = len(selected)
        self.valid_indices = [
            i for i, l in enumerate(all_labels)
            if l in selected
        ]
        self.valid_labels  = [
            self.id_to_new[all_labels[i]]
            for i in self.valid_indices
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx  = self.valid_indices[idx]
        offset    = self.offsets[real_idx]
        new_label = self.valid_labels[idx]

        with open(self.rec_path, 'rb') as f:
            f.seek(offset)
            header = f.read(8)
            magic, length = struct.unpack('<II', header)
            body = f.read(length)

        img_arr = np.frombuffer(body[24:], dtype=np.uint8)
        img     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)

        if self.transform:
            img = Image.fromarray(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            )
            img = self.transform(img)

        return img, new_label