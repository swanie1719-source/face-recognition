import struct
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


class MS1MStreamDataset(Dataset):
    """
    流式读取 MS1M .rec 文件
    训练时按需解码，不预先解压
    不占额外磁盘空间
    """

    def __init__(self, rec_path, idx_path,
                 num_identities=2000, transform=None, seed=42):
        self.rec_path = rec_path
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
        print(f"数据集就绪: {len(self.valid_indices)} 张图, "
              f"{self.num_classes} 个identity")



    def _detect_label_offset(self):
        """
        自动检测 label 在 body 中的正确偏移量
        原理：找到一个偏移量，使得读出的值
              范围在 [0, 85742] 且有大量重复
        """
        import random
        # 采样1000条记录，测试不同偏移量
        sample_offsets = random.sample(self.offsets,
                                       min(1000, len(self.offsets)))

        candidates = {}  # offset → unique值数量

        with open(self.rec_path, 'rb') as f:
            for offset in sample_offsets:
                f.seek(offset)
                header = f.read(8)
                if len(header) < 8:
                    continue
                magic, length = struct.unpack('<II', header)
                if magic != 0xced7230a or length < 40:
                    continue
                body = f.read(min(length, 40))

                for pos in range(0, 32, 4):
                    if pos + 4 > len(body):
                        continue
                    # 分别尝试 int32 和 float32
                    for fmt, key in [('<i', f'int_{pos}'),
                                     ('<f', f'float_{pos}')]:
                        val = struct.unpack(fmt, body[pos:pos + 4])[0]
                        if fmt == '<f':
                            val = int(val)
                        if key not in candidates:
                            candidates[key] = set()
                        candidates[key].add(val)

        # 找出：unique值在 [50, 90000] 范围内的候选
        # （太少说明是常数，太多说明是序号不是label）
        best = None
        best_score = 0
        for key, unique_vals in candidates.items():
            n = len(unique_vals)
            if 50 <= n <= 90000:
                # 检查值域是否合理
                vals = list(unique_vals)
                if min(vals) >= 0 and max(vals) <= 90000:
                    score = n  # unique值越多越可能是label
                    if score > best_score:
                        best_score = score
                        best = key

        if best is None:
            raise ValueError("无法自动检测label偏移量，请手动指定")

        # 解析出偏移量和格式
        fmt, pos = best.split('_')[0], int(best.split('_')[1])
        self.label_fmt = '<i' if fmt == 'int' else '<f'
        return pos

    def _build_index(self, num_identities, seed):
        import random
        random.seed(seed)
        np.random.seed(seed)

        print("扫描label建立索引...")
        all_labels = []

        with open(self.rec_path, 'rb') as f:
            from tqdm import tqdm
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
                # label在offset=12，float32转int
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
        self.num_classes = len(selected)
        self.valid_indices = [
            i for i, l in enumerate(all_labels)
            if l in selected
        ]
        self.valid_labels = [
            self.id_to_new[all_labels[i]]
            for i in self.valid_indices
        ]


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        按需读取：每次只读一张图片
        """
        real_idx = self.valid_indices[idx]
        offset = self.offsets[real_idx]
        new_label = self.valid_labels[idx]

        with open(self.rec_path, 'rb') as f:
            f.seek(offset)
            header = f.read(8)
            magic, length = struct.unpack('<II', header)
            body = f.read(length)

        img_bytes = body[24:]
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)

        if self.transform:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = self.transform(img)

        return img, new_label