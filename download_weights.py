# download_weights.py
from huggingface_hub import hf_hub_download

# 下载 YOLOv8n-face 权重（WIDERFace训练）
path = hf_hub_download(
    repo_id   = "arnabdhar/YOLOv8-Face-Detection",
    filename  = "model.pt",
    local_dir = "weights/"
)
print(f"权重已下载到: {path}")