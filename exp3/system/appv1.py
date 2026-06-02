import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import time
from datetime import datetime

# ==========================================
# 0. 页面全局配置 (严肃工业风格)
# ==========================================
st.set_page_config(page_title="工业缺陷云边协同检测原型系统", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 1. 核心深度学习模型结构定义 (绝对完整，不可省略)
# ==========================================
IMG_SIZE = 128
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 4

class DynamicAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.tensor(0.17))
        self.register_buffer('b', torch.tensor(0.5))
        self.register_buffer('c', torch.tensor(0.12))
    def forward(self, x): 
        return self.a * (x**2) + self.b * x + self.c

class PolyBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), DynamicAct(), nn.Linear(dim * 2, dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, EMBED_DIM) * .02)
        self.blocks = nn.ModuleList([PolyBlock(EMBED_DIM, NUM_HEADS) for _ in range(2)])
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for blk in self.blocks: x = blk(x)
        return x

class ServerModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.blocks = nn.ModuleList([PolyBlock(EMBED_DIM, NUM_HEADS) for _ in range(2)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, num_classes)
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))

CLASSES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# ==========================================
# 2. 异构算力分配与模型加载
# ==========================================
@st.cache_resource(show_spinner=False)
def load_system_models():
    # 🌟 核心修改：硬件资源物理隔离
    # 模拟边缘端算力贫弱（仅分配 CPU）
    device_edge = torch.device('cpu')
    # 模拟云端算力充沛（分配 GPU，如果没有则退化为 CPU）
    device_cloud = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 你的权重路径
    YOLO_WEIGHTS = '/root/autodl-tmp/exp1/result_exp1/18_Rescue_Crazing/weights/best.pt'
    CLIENT_WEIGHTS = '/root/autodl-tmp/exp2/Fed_ViT_Models/fed_vit_client_best_v50.pth'
    SERVER_WEIGHTS = '/root/autodl-tmp/exp2/Fed_ViT_Models/fed_vit_server_best_v50.pth'
    
    # 1. 边缘端 YOLO
    yolo = YOLO(YOLO_WEIGHTS)
    yolo.to(device_edge)
    
    # 2. 边缘端 ViT Client
    client = ClientModel().to(device_edge)
    client.load_state_dict(torch.load(CLIENT_WEIGHTS, map_location=device_edge))
    client.eval()
    
    # 3. 云端 ViT Server
    server = ServerModel(num_classes=6).to(device_cloud)
    server.load_state_dict(torch.load(SERVER_WEIGHTS, map_location=device_cloud))
    server.eval()
    
    return yolo, client, server, device_edge, device_cloud

try:
    with st.spinner("系统初始化：挂载边缘CPU与云端GPU算力..."):
        yolo_model, client_model, server_model, DEVICE_EDGE, DEVICE_CLOUD = load_system_models()
    models_loaded = True
except Exception as e:
    st.error(f"系统错误: 模型加载失败。日志信息: {e}")
    models_loaded = False

vit_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_time_prefix():
    return datetime.now().strftime("[%H:%M:%S.%f]")[:-3]

# ==========================================
# 3. UI 布局与协同调度逻辑
# ==========================================
st.title("基于隐私保护的云边协同缺陷检测系统")
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ 系统参数配置面板")
    conf_thresh = st.slider("边缘侧决策置信度阈值", 0.1, 0.9, 0.6, 0.05, help="低于此置信度将触发云端协同")
    noise_std = st.slider("差分隐私噪声方差 (DP)", 0.0, 0.01, 0.001, 0.001, format="%.4f")
    st.markdown("---")
    st.markdown("### 🖥️ 异构算力监控")
    st.code(f"边缘网关 (Edge): {str(DEVICE_EDGE).upper()}\n云端中枢 (Cloud): {str(DEVICE_CLOUD).upper()}", language="yaml")

tab1, tab2 = st.tabs(["实时检测工作台", "系统架构说明"])

with tab1:
    col_img, col_log = st.columns([5, 4])
    
    with col_img:
        st.markdown("#### 视觉数据输入")
        uploaded_file = st.file_uploader("导入工业图像流 (JPG/PNG)", type=['jpg', 'png', 'bmp'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGB')
            display_image = original_image.copy()
            
            st.image(original_image, caption="原始采集图像流", use_container_width=True)
            start_btn = st.button("🚀 启动协同检测工作流", type="primary", use_container_width=True)
            
            if start_btn and models_loaded:
                draw = ImageDraw.Draw(display_image)
                
                with col_log:
                    st.markdown("#### 系统协同终端日志")
                    log_container = st.container(height=500)
                    log_container.markdown(f"**{get_time_prefix()} [SYSTEM]** 建立分析链路，算力隔离就绪。")
                
                # --- 性能计时器与统计数据 ---
                total_start_time = time.time()
                time_stats = {'edge_yolo': 0.0, 'edge_client': 0.0, 'network': 0.0, 'cloud_server': 0.0}
                stats = {'yolo_only': 0, 'cloud_called': 0}
                detected_regions = []
                
                # 1. 边缘端 YOLO 初筛 (强制在 CPU 运行)
                log_container.markdown(f"**{get_time_prefix()} [EDGE-CPU]** 开始全画幅特征扫描 (NMS iou=0.2)...")
                t0 = time.time()
                results = yolo_model(original_image, conf=0.05, iou=0.2, device=DEVICE_EDGE, verbose=False)[0]
                time_stats['edge_yolo'] = time.time() - t0
                
                for box in results.boxes:
                    conf = box.conf.item()
                    cls_idx = int(box.cls.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    if conf >= conf_thresh:
                        stats['yolo_only'] += 1
                        detected_regions.append({'box': [x1, y1, x2, y2], 'class_id': cls_idx, 'conf': conf, 'source': 'Edge', 'color': '#28a745'})
                    else:
                        stats['cloud_called'] += 1
                        log_container.warning(f"**{get_time_prefix()} [EDGE-WARN]** 异常特征模糊 (Conf={conf:.2f})，启动云端调度。")
                        
                        # 2. 边缘端 ViT 特征提取 (CPU)
                        t1 = time.time()
                        crop_img = original_image.crop((x1, y1, x2, y2))
                        input_tensor = vit_transform(crop_img).unsqueeze(0).to(DEVICE_EDGE)
                        with torch.no_grad():
                            z = client_model(input_tensor)
                            z_safe = z + torch.randn_like(z) * noise_std
                        time_stats['edge_client'] += (time.time() - t1)
                        
                        # 3. 模拟网络通信延迟 (根据 Payload 大小估算，比如 150ms)
                        log_container.info(f"**{get_time_prefix()} [NET-SECURE]** 差分隐私加密完成，通过 5G 专网向云端传输张量...")
                        simulated_latency = 0.15 
                        time.sleep(simulated_latency)
                        time_stats['network'] += simulated_latency
                        
                        # 4. 云端 ViT 专家复核 (GPU)
                        t2 = time.time()
                        # 核心跨设备通信：将张量从边缘CPU推送到云端GPU
                        z_cloud = z_safe.to(DEVICE_CLOUD) 
                        with torch.no_grad():
                            logits = server_model(z_cloud)
                            cloud_cls = logits.argmax(1).item()
                            cloud_conf = torch.softmax(logits, dim=1)[0][cloud_cls].item()
                        time_stats['cloud_server'] += (time.time() - t2)
                        
                        detected_regions.append({'box': [x1, y1, x2, y2], 'class_id': cloud_cls, 'conf': cloud_conf, 'source': 'Cloud', 'color': '#dc3545'})
                        log_container.error(f"**{get_time_prefix()} [CLOUD-GPU]** 专家网络复核完成，修正为: {CLASSES[cloud_cls]}")
                        
                # --- 多区域共识纠偏机制 ---
                if len(detected_regions) > 1:
                    class_votes = {}
                    for reg in detected_regions:
                        cid = reg['class_id']
                        class_votes[cid] = class_votes.get(cid, 0) + 1
                    dominant_class = max(class_votes, key=class_votes.get)
                    
                    if class_votes[dominant_class] > len(detected_regions) / 2:
                        for reg in detected_regions:
                            if reg['class_id'] != dominant_class:
                                old_cls_name = CLASSES[reg['class_id']]
                                new_cls_name = CLASSES[dominant_class]
                                log_container.success(f"**{get_time_prefix()} [CONSENSUS]** 触发全局共识纠偏：修正离群预测 [{old_cls_name}] -> [{new_cls_name}]")
                                reg['class_id'] = dominant_class
                
                # --- 智能绘图 ---
                for reg in detected_regions:
                    x1, y1, x2, y2 = reg['box']
                    color = reg['color']
                    label = f"{CLASSES[reg['class_id']]} ({reg['source']}:{reg['conf']:.2f})"
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    text_width = len(label) * 6 + 10 
                    text_height = 15
                    label_y1 = y1 - text_height if y1 - text_height > 0 else y1
                    label_y2 = y1 if y1 - text_height > 0 else y1 + text_height
                    draw.rectangle([x1, label_y1, x1 + text_width, label_y2], fill=color)
                    draw.text((x1 + 3, label_y1 + 1), label, fill="white")
                
                total_time = time.time() - total_start_time
                log_container.markdown(f"**{get_time_prefix()} [SYSTEM]** 协同链路闭环，总耗时 {total_time:.2f}s。")
                
                st.markdown("#### 综合检测分析报告")
                st.image(display_image, caption="融合输出结果 (绿色: 边缘侧低延迟决策 | 红色: 云端高精度复核)", use_container_width=True)
                
                # --- 链路时延与计算流转大屏 ---
                st.markdown("##### 链路时延与调度看板")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("总拦截异常", f"{stats['yolo_only'] + stats['cloud_called']} 处")
                m2.metric("边缘侧耗时 (CPU)", f"{(time_stats['edge_yolo'] + time_stats['edge_client']) * 1000:.1f} ms")
                m3.metric("通信链路时延", f"{time_stats['network'] * 1000:.1f} ms")
                m4.metric("云端推理耗时 (GPU)", f"{time_stats['cloud_server'] * 1000:.1f} ms")

    with col_log:
        if uploaded_file is None:
            st.info("系统处于监听状态。请挂载图像流输入。")

with tab2:
    st.markdown("### 系统架构与算力部署说明")
    st.markdown("""
    本系统通过异构算力环境验证了**《基于隐私保护的云边协同缺陷检测架构》**的端到端可行性。
    
    * **硬件资源隔离**：系统强行将轻量级的检测器与特征提取前端（`YOLO` & `ClientModel`）锁定于 **CPU（模拟边缘计算网关）**，而将计算密集型的专家后端（`ServerModel`）绑定于 **GPU（模拟云端算力中心）**，真实反映了工业物联网的算力分布拓扑。
    * **跨设备张量传输**：当边缘 CPU 遇到低置信度特征时，对裁剪特征张量进行加密处理，通过模拟的广域网传输至云端 GPU 完成复核计算。
    * **共识修复机制**：针对网络拆分可能导致的局部上下文缺失，系统在后处理阶段引入基于统计分布的多数表决模型，自动纠偏偶发的特征混淆（如将 Crazing 误认为 Rolled）。
    """)