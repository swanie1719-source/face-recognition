import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pickle
import struct
from tqdm import tqdm
from PIL import Image
import cv2
from data.ms1m_dataset import MS1MStreamDataset

# ── 路径配置 ──────────────────────────────────────────
MS1M_REC   = "/kaggle/input/datasets/thnhnguyntrng/faces-ms1mrefinev2-112x112/faces_emore/train.rec"
MS1M_IDX   = "/kaggle/input/datasets/thnhnguyntrng/faces-ms1mrefinev2-112x112/faces_emore/train.idx"
LFW_BIN    = "/kaggle/input/datasets/thnhnguyntrng/faces-ms1mrefinev2-112x112/faces_emore/lfw.bin"
CKPT_DIR   = "/kaggle/working/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 超参数 ────────────────────────────────────────────
CFG = {
    'num_identities' : 2000,
    'embedding_dim'  : 512,
    'batch_size'     : 64,
    'epochs'         : 30,
    'lr'             : 0.1,
    'weight_decay'   : 5e-4,
    'seed'           : 42,
    'save_every'     : 5,       # 每5个epoch保存一次
    'num_workers'    : 2,
}

# ── LFW 验证集加载 ────────────────────────────────────
def load_lfw_bin(bin_path):
    """加载 lfw.bin 验证集"""
    with open(bin_path, 'rb') as f:
        bins, labels = pickle.load(f, encoding='bytes')
    return bins, labels


def evaluate_lfw(model, bins, labels, device, batch_size=64):
    """
    在 LFW 上评估模型
    计算 TAR@FAR 和准确率
    """
    model.eval()
    features = []

    with torch.no_grad():
        for i in range(0, len(bins), batch_size):
            batch_bins = bins[i:i+batch_size]
            imgs = []
            for b in batch_bins:
                img_arr = np.frombuffer(b, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if img is None:
                    img = np.zeros((112, 112, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.tensor(img).permute(2, 0, 1).float()
                img = (img / 255.0 - 0.5) / 0.5
                imgs.append(img)

            batch = torch.stack(imgs).to(device)
            feat  = model(batch)
            features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels   = np.array(labels)

    # 计算每对图片的余弦相似度
    scores = []
    for i in range(0, len(labels), 2):
        if i + 1 >= len(features):
            break
        f1 = features[i]
        f2 = features[i+1]
        score = np.dot(f1, f2)
        scores.append(score)

    scores = np.array(scores)
    pairs  = labels[:len(scores)*2:2] if len(labels) > 0 else labels

    # 找最优阈值
    best_acc = 0
    best_thr = 0
    for thr in np.arange(-1, 1, 0.01):
        preds = (scores > thr).astype(int)
        acc   = (preds == pairs[:len(preds)]).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_acc, best_thr


# ── 数据集（复用之前的MS1MStreamDataset）────────────
class MS1MStreamDataset(torch.utils.data.Dataset):
    def __init__(self, rec_path, idx_path,
                 num_identities=2000, transform=None, seed=42):
        import random
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
        import random
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
                val = struct.unpack('<f', data[12:16])[0]
                all_labels.append(int(val))

        valid_ids = list(set(l for l in all_labels if l >= 0))
        selected  = set(random.sample(
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
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = self.transform(img)

        return img, new_label


# ── 主训练函数 ────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    # 数据集
    from data.augmentation import TrainTransform
    dataset = MS1MStreamDataset(
        rec_path       = MS1M_REC,
        idx_path       = MS1M_IDX,
        num_identities = CFG['num_identities'],
        transform      = TrainTransform(face_size=112),
        seed           = CFG['seed']
    )
    loader = DataLoader(
        dataset,
        batch_size  = CFG['batch_size'],
        shuffle     = True,
        num_workers = CFG['num_workers'],
        pin_memory  = True
    )

    # 模型
    from models.recognizer import FaceModel
    model = FaceModel(
        num_classes   = dataset.num_classes,
        embedding_dim = CFG['embedding_dim'],
        pretrained    = True
    ).to(device)

    # 优化器和学习率调度
    optimizer = SGD(
        model.parameters(),
        lr           = CFG['lr'],
        momentum     = 0.9,
        weight_decay = CFG['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = CFG['epochs'],
        eta_min = 1e-5
    )

    # 加载 LFW 验证集
    print("加载LFW验证集...")
    lfw_bins, lfw_labels = load_lfw_bin(LFW_BIN)
    print(f"LFW: {len(lfw_bins)} 张图片")

    # 训练循环
    best_acc = 0
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        pbar = tqdm(loader,
                    desc=f"Epoch {epoch}/{CFG['epochs']}")
        for imgs, labels in pbar:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features, loss = model(imgs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total      += labels.size(0)
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr'  : f"{scheduler.get_last_lr()[0]:.6f}"
            })

        scheduler.step()
        avg_loss = total_loss / len(loader)

        # LFW 验证
        lfw_acc, lfw_thr = evaluate_lfw(
            model.recognizer, lfw_bins, lfw_labels, device
        )
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, "
              f"LFW={lfw_acc*100:.2f}%, thr={lfw_thr:.3f}")

        # 保存最优模型
        if lfw_acc > best_acc:
            best_acc = lfw_acc
            torch.save({
                'epoch'     : epoch,
                'state_dict': model.recognizer.state_dict(),
                'lfw_acc'   : lfw_acc,
                'cfg'       : CFG,
            }, os.path.join(CKPT_DIR, 'best_model.pth'))
            print(f"  ★ 最优模型已保存 LFW={best_acc*100:.2f}%")

        # 定期保存
        if epoch % CFG['save_every'] == 0:
            torch.save({
                'epoch'     : epoch,
                'state_dict': model.recognizer.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'lfw_acc'   : lfw_acc,
            }, os.path.join(CKPT_DIR, f'epoch_{epoch}.pth'))

    print(f"\n训练完成！最优 LFW 准确率: {best_acc*100:.2f}%")


if __name__ == '__main__':
    train()