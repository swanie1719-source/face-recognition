import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
sys.path.insert(0, '.')

from inference import FaceRecognitionSystem


# ── 页面配置 ──────────────────────────────────────────
st.set_page_config(
    page_title = "人脸识别系统",
    page_icon  = "👤",
    layout     = "wide"
)

st.title("👤 基于 YOLOv12 的人脸识别系统")
st.caption("ResNet50 + ArcFace | LFW 准确率 94.72%")


# ── 初始化系统（只加载一次）──────────────────────────
@st.cache_resource
def load_system():
    return FaceRecognitionSystem(
        detector_weights = "weights/model.pt",
        db_path          = "face_db/",
        conf_threshold   = 0.5,
        sim_threshold    = 0.35
    )


system = load_system()


# ── 侧边栏：功能选择 ──────────────────────────────────
st.sidebar.title("功能菜单")
mode = st.sidebar.radio(
    "选择功能",
    ["人脸识别", "人脸注册", "数据库管理", "系统信息"]
)

# 侧边栏参数调节
st.sidebar.divider()
st.sidebar.subheader("参数设置")
sim_threshold = st.sidebar.slider(
    "相似度阈值", 0.1, 0.9, 0.4, 0.05,
    help="低于此值判定为陌生人"
)
system.db.threshold = sim_threshold


# ══════════════════════════════════════════════════════
# 功能1：人脸识别
# ══════════════════════════════════════════════════════
if mode == "人脸识别":
    st.header("人脸识别")

    input_type = st.radio(
        "输入方式",
        ["上传图片", "摄像头拍照"],
        horizontal=True
    )

    if input_type == "上传图片":
        uploaded = st.file_uploader(
            "上传图片",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            # 读取图片
            pil_img = Image.open(uploaded).convert("RGB")
            img_arr = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("原始图片")
                st.image(pil_img, use_container_width=True)

            with st.spinner("识别中..."):
                results = system.recognize(img_bgr)
                vis_bgr = system.visualize(img_bgr, results)
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("识别结果")
                st.image(vis_rgb, use_container_width=True)

            # 显示详细结果
            st.divider()
            if results:
                st.subheader(f"检测到 {len(results)} 张人脸")
                for i, res in enumerate(results):
                    col_face, col_info = st.columns([1, 3])
                    with col_face:
                        face_rgb = cv2.cvtColor(
                            res['face'], cv2.COLOR_BGR2RGB
                        )
                        st.image(face_rgb, width=100)
                    with col_info:
                        name = res['name']
                        sim  = res['similarity']
                        if name == "陌生人":
                            st.error(f"陌生人 (相似度: {sim:.3f})")
                        else:
                            st.success(f"**{name}** (相似度: {sim:.3f})")
                        st.caption(
                            f"检测置信度: {res['confidence']:.3f}"
                        )
            else:
                st.warning("未检测到人脸，请换一张图片")

    else:  # 摄像头拍照
        img_file = st.camera_input("拍照识别")
        if img_file:
            pil_img = Image.open(img_file).convert("RGB")
            img_arr = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

            with st.spinner("识别中..."):
                results = system.recognize(img_bgr)
                vis_bgr = system.visualize(img_bgr, results)
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

            st.image(vis_rgb, use_container_width=True)

            if results:
                for res in results:
                    name = res['name']
                    sim  = res['similarity']
                    if name == "陌生人":
                        st.error(f"陌生人 (相似度: {sim:.3f})")
                    else:
                        st.success(f"识别为：**{name}** "
                                   f"(相似度: {sim:.3f})")
            else:
                st.warning("未检测到人脸")


# ══════════════════════════════════════════════════════
# 功能2：人脸注册
# ══════════════════════════════════════════════════════
elif mode == "人脸注册":
    st.header("人脸注册")
    st.info("注册后无需重新训练，立即生效")

    name = st.text_input("请输入姓名", placeholder="例如：张三")

    input_type = st.radio(
        "图片来源",
        ["上传照片", "摄像头拍照"],
        horizontal=True
    )

    img_bgr = None

    if input_type == "上传照片":
        uploaded = st.file_uploader(
            "上传清晰正面照",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            img_arr = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            st.image(pil_img, width=300)
    else:
        img_file = st.camera_input("拍摄注册照")
        if img_file:
            pil_img = Image.open(img_file).convert("RGB")
            img_arr = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    if st.button("确认注册", type="primary"):
        if not name:
            st.error("请先输入姓名")
        elif img_bgr is None:
            st.error("请先提供照片")
        else:
            with st.spinner("注册中..."):
                success, msg = system.register(img_bgr, name)
            if success:
                st.success(msg)
                st.balloons()
            else:
                st.error(msg)


# ══════════════════════════════════════════════════════
# 功能3：数据库管理
# ══════════════════════════════════════════════════════
elif mode == "数据库管理":
    st.header("人脸数据库管理")

    total   = system.db.index.ntotal
    persons = len(set(system.db.names)) if system.db.names else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("总记录数", total)
    col2.metric("人物数量", persons)
    col3.metric("相似度阈值", f"{sim_threshold:.2f}")

    if system.db.names:
        st.divider()
        st.subheader("已注册人员")
        from collections import Counter

        counter = Counter(system.db.names)

        for name, count in sorted(counter.items()):
            col_name, col_count, col_edit = st.columns([3, 1, 2])
            col_name.write(f"**{name}**")
            col_count.write(f"{count} 张")

            # 重命名功能
            with col_edit:
                new_name = st.text_input(
                    "改名",
                    value=name,
                    key=f"rename_{name}",
                    label_visibility="collapsed"
                )
                if st.button("确认", key=f"btn_{name}"):
                    if new_name and new_name != name:
                        # 把所有这个人的名字替换掉
                        system.db.names = [
                            new_name if n == name else n
                            for n in system.db.names
                        ]
                        system.db.save(system.db_path)
                        st.success(f"{name} → {new_name}")
                        st.rerun()

    st.divider()
    if st.button("清空数据库", type="secondary"):
        import faiss
        system.db.index = faiss.IndexFlatIP(512)
        system.db.names = []
        system.db.save(system.db_path)
        st.success("数据库已清空")
        st.rerun()


# ══════════════════════════════════════════════════════
# 功能4：系统信息
# ══════════════════════════════════════════════════════
elif mode == "系统信息":
    st.header("系统信息")

    import torch
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("模型配置")
        st.json({
            "检测模型"  : "YOLOv8n-face",
            "识别模型"  : "ResNet50 + ArcFace",
            "特征维度"  : 512,
            "LFW准确率" : "94.72%",
            "训练数据"  : "MS1MV3 (2000 identities)",
            "推理设备"  : system.device
        })

    with col2:
        st.subheader("系统状态")
        st.json({
            "数据库记录数" : system.db.index.ntotal,
            "注册人数"    : len(set(system.db.names)),
            "相似度阈值"  : sim_threshold,
            "GPU可用"    : torch.cuda.is_available()
        })

    st.divider()
    st.subheader("技术栈")
    st.markdown("""
    | 模块 | 技术 |
    |------|------|
    | 人脸检测 | YOLOv12 (ultralytics) |
    | 特征提取 | ResNet50 |
    | 损失函数 | ArcFace |
    | 特征检索 | FAISS |
    | 前端框架 | Streamlit |
    | 深度学习 | PyTorch |
    """)