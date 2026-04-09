import os
import cv2
import numpy as np
import pickle
import faiss
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image


class FaceDatabase:
    def __init__(self, feature_dim=512, threshold=0.4):
        """
        feature_dim : 特征维度
        threshold   : 余弦相似度阈值，低于此值判定为陌生人
        """
        self.feature_dim = feature_dim
        self.threshold   = threshold
        self.index       = faiss.IndexFlatIP(feature_dim)
        self.names       = []
        self.image_paths = []

    # ── 提取特征 ───────────────────────────────────────
    def extract_feature(self, model, image, device='cpu'):
        """
        model : FaceRecognizer 模型（eval模式）
        image : PIL Image 或 numpy array BGR
        返回  : 512维归一化特征向量
        """
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std =[0.5, 0.5, 0.5]
            )
        ])

        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        tensor  = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(tensor)

        feature = feature.cpu().numpy()[0]
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature

    # ── 注册单张图片 ───────────────────────────────────
    def register_one(self, model, image, name, device='cpu'):
        """
        向数据库注册一张人脸
        image: 图片路径 或 PIL Image 或 numpy array
        """
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image

        feature = self.extract_feature(model, img, device)
        self.index.add(feature.reshape(1, -1).astype('float32'))
        self.names.append(name)
        print(f"已注册：{name}")
        return feature

    # ── 批量从目录入库 ─────────────────────────────────
    def build_from_directory(self, model, root_dir, device='cpu'):
        """
        从按人名分类的目录批量入库
        目录结构：
            root_dir/
                张三/
                    img1.jpg
                李四/
                    img1.jpg
        """
        root        = Path(root_dir)
        person_dirs = sorted([d for d in root.iterdir()
                              if d.is_dir()])
        print(f"发现 {len(person_dirs)} 个人，开始入库...")

        for person_dir in tqdm(person_dirs):
            name      = person_dir.name
            img_files = (list(person_dir.glob("*.jpg")) +
                         list(person_dir.glob("*.png")))

            for img_path in img_files:
                try:
                    img     = Image.open(img_path).convert('RGB')
                    feature = self.extract_feature(
                        model, img, device
                    )
                    self.index.add(
                        feature.reshape(1, -1).astype('float32')
                    )
                    self.names.append(name)
                    self.image_paths.append(str(img_path))
                except Exception as e:
                    print(f"跳过 {img_path}: {e}")

        print(f"入库完成，共 {self.index.ntotal} 条记录")

    # ── 查询识别 ───────────────────────────────────────
    def search(self, feature, top_k=1):
        """
        直接用特征向量检索
        feature: numpy array (512,)
        返回: list of (name, similarity_score)
        """
        if self.index.ntotal == 0:
            return [("数据库为空", 0.0)]

        query = feature.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            name = self.names[idx]
            if score >= self.threshold:
                results.append((name, float(score)))
            else:
                results.append(("陌生人", float(score)))

        return results

    def identify(self, model=None, image=None,
                 device='cpu', feature=None):
        """
        识别人脸，返回最可能的名字和置信度
        支持直接传入 feature 向量（跳过模型推理）
        """
        if feature is None:
            if model is None or image is None:
                return "参数错误", 0.0
            feature = self.extract_feature(model, image, device)

        results = self.search(feature, top_k=1)
        if results:
            return results[0]
        return "识别失败", 0.0

    # ── 保存数据库 ─────────────────────────────────────
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(save_dir, "face_index.faiss")
        )

        meta = {
            'names'       : self.names,
            'image_paths' : self.image_paths,
            'feature_dim' : self.feature_dim,
            'threshold'   : self.threshold,
        }
        with open(os.path.join(save_dir, "meta.pkl"), 'wb') as f:
            pickle.dump(meta, f)

        print(f"数据库已保存: {save_dir} "
              f"({self.index.ntotal}条, "
              f"{len(set(self.names))}人)")

    # ── 加载数据库 ─────────────────────────────────────
    @classmethod
    def load(cls, save_dir):
        meta_path = os.path.join(save_dir, "meta.pkl")
        idx_path  = os.path.join(save_dir, "face_index.faiss")

        if not os.path.exists(meta_path):
            print("数据库文件不存在，创建新数据库")
            return cls()

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        db             = cls(
            feature_dim = meta['feature_dim'],
            threshold   = meta['threshold']
        )
        db.names       = meta['names']
        db.image_paths = meta['image_paths']
        db.index       = faiss.read_index(idx_path)

        print(f"数据库加载成功: {db.index.ntotal} 条记录, "
              f"{len(set(db.names))} 人")
        return db

    # ── 统计信息 ───────────────────────────────────────
    def get_stats(self):
        from collections import Counter
        counter = Counter(self.names)
        return {
            'total'         : self.index.ntotal,
            'num_persons'   : len(counter),
            'avg_per_person': (self.index.ntotal /
                               max(len(counter), 1)),
            'persons'       : dict(counter)
        }