# test_detector.py 更新版
import cv2
import sys
sys.path.insert(0, '.')
from models.detector import FaceDetector

detector = FaceDetector(
    model_path    = "weights/model.pt",
    conf_threshold = 0.5
)

# 换成你本地一张有人脸的照片路径
img = cv2.imread("testphoto/006aauYzgy1hzja0cgg8lj311y1kwgrf.jpg")
detections = detector.detect_and_align(img)

print(f"检测到 {len(detections)} 个人脸")
for i, det in enumerate(detections):
    print(f"  人脸{i}: bbox={det['bbox']}, 置信度={det['confidence']:.3f}")
    print(f"         对齐后尺寸={det['face'].shape}")

# 保存可视化结果
vis = detector.visualize(img, detections)
cv2.imwrite("detection_result.jpg", vis)
print("可视化结果已保存到 detection_result.jpg")