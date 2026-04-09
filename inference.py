import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from models.detector import FaceDetector
from models.recognizer import FaceRecognizer
from data.preprocess import FacePreprocessor
from build_database import FaceDatabase


class FaceRecognitionSystem:
    """
    完整人脸识别系统
    整合检测、对齐、特征提取、数据库检索
    """

    def __init__(self,
                 detector_weights  = "weights/model.pt",
                 recognizer_weights = "weights/best_model.pth",
                 db_path           = "face_db/",
                 conf_threshold    = 0.5,
                 sim_threshold     = 0.4,
                 device            = None):

        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"推理设备: {self.device}")

        # 初始化检测器
        print("加载人脸检测模型...")
        self.detector = FaceDetector(
            model_path     = detector_weights,
            conf_threshold = conf_threshold,
            device         = self.device
        )

        # 初始化特征提取器
        print("加载人脸识别模型...")
        self.recognizer = FaceRecognizer(
            embedding_dim = 512,
            pretrained    = False
        ).to(self.device)

        ckpt = torch.load(
            recognizer_weights,
            map_location=self.device,
            weights_only=False
        )
        self.recognizer.load_state_dict(ckpt['state_dict'])
        self.recognizer.eval()
        print(f"模型加载成功，训练时LFW: {ckpt.get('lfw_acc', 0)*100:.2f}%")

        # 初始化预处理器
        self.preprocessor = FacePreprocessor(face_size=112)

        # 初始化数据库
        self.db = FaceDatabase(
            feature_dim = 512,
            threshold   = sim_threshold
        )

        # 加载已有数据库
        if Path(db_path).exists():
            self.db = FaceDatabase.load(db_path)
            print(f"数据库加载成功: {self.db.index.ntotal} 条记录")
        else:
            print("数据库为空，请先注册人脸")

        self.db_path = db_path

    # ── 提取单张人脸特征 ──────────────────────────────
    def extract_feature(self, face_img):
        """
        face_img: numpy array BGR (112x112)
        返回: numpy array (512,) 归一化特征向量
        """
        tensor = self.preprocessor.process(face_img)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.recognizer(tensor)

        feature = feature.cpu().numpy()[0]
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature

    # ── 注册新人脸 ────────────────────────────────────
    def register(self, image, name):
        """
        image: 图片路径 或 numpy array BGR
        name:  人物姓名
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        # 检测人脸
        detections = self.detector.detect_and_align(img)
        if not detections:
            return False, "未检测到人脸"

        # 取置信度最高的人脸
        best = max(detections, key=lambda x: x['confidence'])
        face = best['face']

        # 提取特征并入库
        feature = self.extract_feature(face)
        import faiss
        self.db.index.add(
            feature.reshape(1, -1).astype('float32')
        )
        self.db.names.append(name)

        # 保存数据库
        self.db.save(self.db_path)
        return True, f"注册成功: {name}"

    # ── 识别单张图片 ──────────────────────────────────
    def recognize(self, image):
        """
        识别图片中的所有人脸
        返回: list of dict
          {
            'name'      : 姓名或"陌生人"
            'similarity': 相似度分数
            'bbox'      : 边界框
            'face'      : 对齐后人脸图片
            'confidence': 检测置信度
          }
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        detections = self.detector.detect_and_align(img)
        results    = []

        for det in detections:
            face    = det['face']
            feature = self.extract_feature(face)

            # 数据库检索
            name, similarity = self.db.identify(
                model  = None,
                image  = None,
                device = self.device
            )

            # 直接用feature检索（绕过db.identify的模型调用）
            if self.db.index.ntotal > 0:
                query = feature.reshape(1, -1).astype('float32')
                scores, indices = self.db.index.search(query, 1)
                score = float(scores[0][0])
                idx   = int(indices[0][0])

                if idx >= 0 and score >= self.db.threshold:
                    name = self.db.names[idx]
                else:
                    name = "陌生人"
                similarity = score
            else:
                name       = "数据库为空"
                similarity = 0.0

            results.append({
                'name'      : name,
                'similarity': similarity,
                'bbox'      : det['bbox'],
                'face'      : face,
                'confidence': det['confidence']
            })

        return results

    # ── 在图片上画结果 ────────────────────────────────
    def visualize(self, image, results):
        """画出识别结果"""
        vis = image.copy()

        for res in results:
            x1, y1, x2, y2 = res['bbox']
            name       = res['name']
            similarity = res['similarity']
            is_stranger = name == "陌生人"

            # 框的颜色：认识的人绿色，陌生人红色
            color = (0, 0, 255) if is_stranger else (0, 255, 0)

            # 画框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 标签
            label = f"{name} {similarity:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )[0]

            # 标签背景
            cv2.rectangle(
                vis,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0], y1),
                color, -1
            )

            # 标签文字
            cv2.putText(
                vis, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )

        return vis

    # ── 实时摄像头识别 ────────────────────────────────
    def run_camera(self, camera_id=0):
        """
        实时摄像头人脸识别
        按 Q 退出
        """
        cap = cv2.VideoCapture(camera_id)
        print("摄像头已启动，按 Q 退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.recognize(frame)
            vis     = self.visualize(frame, results)

            # 显示 FPS
            cv2.imshow("人脸识别系统", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()