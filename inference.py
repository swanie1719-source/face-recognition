import cv2
import numpy as np
import torch
from pathlib import Path
from insightface.app import FaceAnalysis

from models.detector import FaceDetector
from build_database import FaceDatabase


class FaceRecognitionSystem:
    """
    完整人脸识别系统
    检测：YOLOv12（ultralytics）
    特征提取：InsightFace buffalo_l（w600k_r50）
    识别：FAISS 余弦相似度检索
    """

    def __init__(self,
                 detector_weights = "weights/model.pt",
                 db_path          = "face_db/",
                 conf_threshold   = 0.5,
                 sim_threshold    = 0.35):

        print("加载人脸检测模型...")
        self.detector = FaceDetector(
            model_path     = detector_weights,
            conf_threshold = conf_threshold,
        )

        print("加载 InsightFace 特征提取模型...")
        self.insight = FaceAnalysis(name='buffalo_l')
        self.insight.prepare(ctx_id=-1)
        print("InsightFace 加载成功")

        # 初始化数据库
        self.db = FaceDatabase(
            feature_dim = 512,
            threshold   = sim_threshold
        )
        if Path(db_path).exists():
            self.db = FaceDatabase.load(db_path)
            print(f"数据库加载成功: "
                  f"{self.db.index.ntotal} 条记录")
        else:
            print("数据库为空，请先注册人脸")

        self.db_path = db_path

    # ── 提取特征 ───────────────────────────────────────
    def extract_feature(self, img):
        """
        img: numpy array BGR（已对齐的112×112人脸）
        直接用 recognition 模型提取，跳过检测步骤
        """
        # 找到 recognition 模型
        recog_model = None
        for name, model in self.insight.models.items():
            if 'recognition' in name or 'arcface' in name.lower():
                recog_model = model
                break

        if recog_model is None:
            return None

        # 直接提取特征，不经过检测
        feat = recog_model.get_feat(img)
        if feat is None:
            return None

        feat = feat.flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat

    # ── 注册人脸 ───────────────────────────────────────
    def register(self, image, name):
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        # 先用 YOLOv12 检测对齐，和识别时保持一致
        detections = self.detector.detect_and_align(img)
        if not detections:
            return False, "未检测到人脸"

        # 取置信度最高的人脸
        best = max(detections, key=lambda x: x['confidence'])
        face = best['face']  # 已对齐的112×112人脸

        # 用对齐后的人脸提取特征
        feat = self.extract_feature(face)
        if feat is None:
            return False, "特征提取失败"

        import faiss
        self.db.index.add(
            feat.reshape(1, -1).astype('float32')
        )
        self.db.names.append(name)
        self.db.save(self.db_path)
        return True, f"注册成功: {name}"

    # ── 识别 ───────────────────────────────────────────
    def recognize(self, image):
        """
        识别图片中所有人脸
        返回: list of dict
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        # YOLOv12 检测人脸位置
        detections = self.detector.detect_and_align(img)
        results    = []

        for det in detections:
            face = det['face']  # 已对齐的112×112人脸

            # InsightFace 提取特征
            feat = self.extract_feature(face)

            if feat is None or self.db.index.ntotal == 0:
                name       = "未识别"
                similarity = 0.0
            else:
                # FAISS 检索
                query = feat.reshape(1, -1).astype('float32')
                scores, indices = self.db.index.search(query, 1)
                score = float(scores[0][0])
                idx   = int(indices[0][0])

                if idx >= 0 and score >= self.db.threshold:
                    name = self.db.names[idx]
                else:
                    name = "陌生人"
                similarity = score

            results.append({
                'name'      : name,
                'similarity': similarity,
                'bbox'      : det['bbox'],
                'face'      : face,
                'confidence': det['confidence']
            })

        return results

    # ── 可视化 ─────────────────────────────────────────
    def visualize(self, image, results):
        vis = image.copy()

        for res in results:
            x1, y1, x2, y2 = res['bbox']
            name       = res['name']
            similarity = res['similarity']
            is_stranger = name in ["陌生人", "未识别"]
            color = (0, 0, 255) if is_stranger else (0, 255, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {similarity:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )[0]
            cv2.rectangle(
                vis,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0], y1),
                color, -1
            )
            cv2.putText(
                vis, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )

        return vis

    # ── 摄像头实时识别 ─────────────────────────────────
    def run_camera(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        print("摄像头已启动，按 Q 退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.recognize(frame)
            vis     = self.visualize(frame, results)
            cv2.imshow("人脸识别系统", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()