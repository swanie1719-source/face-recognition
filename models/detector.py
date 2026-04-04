import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path


class FaceDetector:
    """
    基于 YOLOv12 的人脸检测器
    功能：
      1. 检测图片中所有人脸，返回边界框
      2. 输出5点关键点（左眼/右眼/鼻尖/左嘴角/右嘴角）
      3. 人脸对齐（仿射变换，把歪脸转正）
    """

    # 标准5点关键点位置（112x112 对齐目标）
    REFERENCE_POINTS = np.array([
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041],  # 右嘴角
    ], dtype=np.float32)

    def __init__(self, model_path=None,
                 conf_threshold=0.5,
                 iou_threshold=0.45,
                 face_size=112,
                 device=None):
        """
        model_path    : YOLOv12 权重路径，None则自动下载
        conf_threshold: 置信度阈值，低于此值的检测框丢弃
        iou_threshold : NMS的IOU阈值
        face_size     : 对齐后的人脸尺寸
        device        : 'cpu' / 'cuda' / None(自动选择)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.face_size = face_size

        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"FaceDetector 使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        if model_path and Path(model_path).exists():
            print(f"加载本地权重: {model_path}")
            model = YOLO(model_path)
        else:
            # 默认加载人脸检测权重
            default = Path("weights/model.pt")
            if default.exists():
                print("加载人脸检测权重...")
                model = YOLO(str(default))
            else:
                print("未找到人脸权重，使用通用权重（精度较低）")
                model = YOLO("yolov8n.pt")
        return model

    # ── 核心检测函数 ───────────────────────────────────
    def detect(self, img):
        """
        检测图片中的所有人脸

        输入：numpy array BGR 或 RGB
        输出：list of dict，每个dict包含：
          {
            'bbox'      : [x1, y1, x2, y2],  # 边界框
            'confidence': float,               # 置信度
            'keypoints' : np.array(5,2),       # 5个关键点坐标
          }
        """
        results = self.model(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                detection = {
                    'bbox': [int(x1), int(y1),
                             int(x2), int(y2)],
                    'confidence': conf,
                    'keypoints': None
                }

                # 提取关键点（如果模型支持）
                if result.keypoints is not None:
                    kpts = result.keypoints.xy[i].cpu().numpy()
                    if len(kpts) >= 5:
                        detection['keypoints'] = kpts[:5]

                detections.append(detection)

        return detections

    # ── 人脸对齐 ───────────────────────────────────────
    def align_face(self, img, keypoints):
        """
        根据5点关键点做仿射变换，把人脸对齐到标准位置

        输入：
          img       : 原图 numpy array BGR
          keypoints : 5个关键点坐标 np.array(5,2)
        输出：
          对齐后的人脸图片 numpy array BGR (face_size x face_size)
        """
        # 目标关键点位置（标准112x112坐标）
        dst_pts = self.REFERENCE_POINTS.copy()
        src_pts = keypoints.astype(np.float32)

        # 计算仿射变换矩阵（用最小二乘法）
        M, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.LMEDS
        )

        if M is None:
            # 变换矩阵计算失败，直接resize
            h, w = img.shape[:2]
            x1 = max(0, int(src_pts[:, 0].min()))
            y1 = max(0, int(src_pts[:, 1].min()))
            x2 = min(w, int(src_pts[:, 0].max()))
            y2 = min(h, int(src_pts[:, 1].max()))
            face = img[y1:y2, x1:x2]
            return cv2.resize(face,
                              (self.face_size, self.face_size))

        # 应用仿射变换
        aligned = cv2.warpAffine(
            img, M,
            (self.face_size, self.face_size),
            borderMode=cv2.BORDER_REFLECT
        )
        return aligned

    # ── 裁剪人脸（无关键点时使用）─────────────────────
    def crop_face(self, img, bbox, margin=0.2):
        """
        根据边界框裁剪人脸，加一点margin避免裁太紧

        margin: 边界框扩展比例，0.2表示扩展20%
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img = img.shape[:2]

        # 计算扩展量
        w = x2 - x1
        h = y2 - y1
        mx = int(w * margin)
        my = int(h * margin)

        # 扩展边界框，不超出图片范围
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w_img, x2 + mx)
        y2 = min(h_img, y2 + my)

        face = img[y1:y2, x1:x2]
        return cv2.resize(face,
                          (self.face_size, self.face_size))

    # ── 一步到位：检测并返回对齐人脸 ──────────────────
    def detect_and_align(self, img):
        """
        检测图片中所有人脸并对齐

        返回：list of dict
          {
            'face'      : np.array (112,112,3) BGR 对齐人脸
            'bbox'      : [x1,y1,x2,y2]
            'confidence': float
          }
        """
        detections = self.detect(img)
        results = []

        for det in detections:
            if det['keypoints'] is not None:
                # 有关键点，做精确对齐
                face = self.align_face(img, det['keypoints'])
            else:
                # 无关键点，直接裁剪
                face = self.crop_face(img, det['bbox'])

            results.append({
                'face': face,
                'bbox': det['bbox'],
                'confidence': det['confidence']
            })

        return results


    def visualize(self, img, detections):
        vis = img.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']

            # 画边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            # 写置信度
            cv2.putText(vis, f"{conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

            # 画关键点（兼容有无关键点两种情况）
            kpts = det.get('keypoints', None)
            if kpts is not None:
                for kpt in kpts:
                    cx, cy = int(kpt[0]), int(kpt[1])
                    cv2.circle(vis, (cx, cy), 3,
                               (0, 0, 255), -1)

        return vis
