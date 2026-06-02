import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw
import time
from datetime import datetime
import platform

# ==========================================
# 0. 页面全局配置与 CSS (恢复你的学术美感)
# ==========================================
st.set_page_config(page_title="端云协同缺陷分析-GPU版", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container { padding: 1rem 2rem 0rem 2rem !important; }
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .academic-log-box {
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 10px;
            background-color: #ffffff;
            font-family: 'Courier New', Courier, monospace;
            font-size: 13px;
            height: 180px;
            overflow-y: auto;
            line-height: 1.6;
            margin-top: 5px;
        }
        .log-time { color: #888888; }
        .tag-info { color: #6c757d; font-weight: bold; }
        .tag-edge { color: #28a745; font-weight: bold; }
        .tag-cloud { color: #007bff; font-weight: bold; }
        h1, h2, h3, h4, h5 { font-weight: 500 !important; color: #333333; margin-bottom: 0.2rem !important;}
    </style>
""", unsafe_allow_html=True)

# 📍 自动获取 CPU 名称
try:
    from cpuinfo import get_cpu_info
    CPU_NAME = get_cpu_info().get('brand_raw', platform.processor())
except:
    CPU_NAME = platform.processor()

# ==========================================
# 1. 核心架构：【GELU 版 + MLP 1024 修复】
# ==========================================
IMG_SIZE, PATCH_SIZE, EMBED_DIM = 128, 16, 256
CLASSES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

class ForwardDefense(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_std = 0.0 
    def forward(self, x):
        if self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

class StandardBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), # 🌟 已修复：对应 1024 维度
            nn.GELU(), 
            nn.Linear(dim * 2, dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))

class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, EMBED_DIM) * .02)
        self.blocks = nn.ModuleList([StandardBlock(EMBED_DIM, 4) for _ in range(2)])
        self.forward_defense = ForwardDefense()
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks: x = blk(x)
        return self.forward_defense(x)

class ServerModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.blocks = nn.ModuleList([StandardBlock(EMBED_DIM, 4) for _ in range(2)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, num_classes)
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x).mean(dim=1))

# ==========================================
# 2. 模型加载 (针对 GPU 平台)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_system():
    device_cloud = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_edge = torch.device('cpu')
    
    # 路径配置
    Y_W = '/root/autodl-tmp/exp1/result_exp1/Final/weights/best.pt'
    C_W = '/root/autodl-tmp/exp3/Dual_Defense_Models/dual_defense_client_best.pth'
    S_W = '/root/autodl-tmp/exp3/Dual_Defense_Models/dual_defense_server_best.pth'
    
    yolo = YOLO(Y_W).to(device_edge)
    client = ClientModel().to(device_edge)
    client.load_state_dict(torch.load(C_W, map_location='cpu'))
    server = ServerModel().to(device_cloud)
    server.load_state_dict(torch.load(S_W, map_location=device_cloud))
    return yolo, client, server, device_edge, device_cloud

try:
    yolo_m, client_m, server_m, D_EDGE, D_CLOUD = load_system()
    models_loaded = True
except Exception as e:
    st.error(f"加载失败: {e}")
    models_loaded = False

# ==========================================
# 3. 布局恢复：顶部状态卡片
# ==========================================
top_col1, top_col2, top_col3, top_col4 = st.columns([4, 1.5, 1.5, 1.5])
with top_col1:
    st.markdown("### 端云协同缺陷分析原型系统 (GPU-Ref)")
with top_col2:
    st.markdown(f"🖥️ **边缘端**: {CPU_NAME[:15]}...") 
with top_col3:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.markdown(f"☁️ **云端侧**: NVIDIA {gpu_name.split(' ')[-1]}")
with top_col4:
    status_placeholder = st.empty()
    status_placeholder.markdown("⚪ **当前状态**: 等待输入")

st.markdown("---")

# 侧边栏
with st.sidebar:
    st.markdown("#### 任务配置")
    up_file = st.file_uploader("输入图像", type=['jpg','png','bmp'], label_visibility="collapsed")
    thr = st.slider("初筛阈值", 0.1, 0.9, 0.6)
    noise_lvl = st.slider("隐私加噪强度", 0.0, 0.05, 0.001, format="%.3f")
    run_btn = st.button("启动协同推理", type="primary", width="stretch")

# ==========================================
# 4. 推理核心逻辑 (Split 逻辑)
# ==========================================
def draw_academic_box(image, box, label, color):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    label_y = max(0, y1 - 14)
    draw.rectangle([x1, label_y, x1 + len(label)*7+8, label_y + 14], fill=color)
    draw.text((x1+3, label_y), label, fill="white")

vit_tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

if up_file:
    img = Image.open(up_file).convert('RGB')
    e_img, c_img = img.copy(), img.copy()
    logs, targets = [], []
    t_total, t_e, t_c = 0.0, 0.0, 0.0

    if run_btn and models_loaded:
        status_placeholder.markdown("🟢 **当前状态**: 运行中...")
        logs.append(f"<span class='log-time'>[{datetime.now().strftime('%H:%M:%S')}]</span> <span class='tag-info'>[INFO]</span> 初始化 GPU 协同链路")
        
        t_start = time.time()
        # 边缘侧处理
        y_res = yolo_m(img, conf=0.05, verbose=False)[0]
        for b in y_res.boxes:
            cf, ci, box = b.conf.item(), int(b.cls.item()), b.xyxy[0].tolist()
            if cf >= thr:
                targets.append({'box': box, 'cls': ci, 'source': 'Edge'})
                draw_academic_box(e_img, box, f"{CLASSES[ci]} {cf:.2f}", "#28a745")
            else:
                client_m.forward_defense.noise_std = noise_lvl
                with torch.no_grad():
                    z = client_m(vit_tf(img.crop(box)).unsqueeze(0))
                targets.append({'box': box, 'tensor': z, 'source': 'Cloud'})
                draw_academic_box(e_img, box, f"Uncertain {cf:.2f}", "#dc3545")
        t_e = (time.time() - t_start) * 1000
        
        # 云端侧处理
        cloud_hit = any(r['source'] == 'Cloud' for r in targets)
        if cloud_hit:
            t1 = time.time()
            for r in targets:
                if r['source'] == 'Cloud':
                    with torch.no_grad():
                        r['cls'] = torch.argmax(server_m(r['tensor'].to(D_CLOUD))).item()
            t_c = (time.time() - t1) * 1000
            logs.append(f"<span class='log-time'>[{datetime.now().strftime('%H:%M:%S')}]</span> <span class='tag-cloud'>[CLOUD]</span> GPU 深度推理完成")
        
        t_total = (time.time() - t_start) * 1000
        status_placeholder.markdown("🔵 **当前状态**: 验证完成")

    # --- 布局恢复：三列图像展示 ---
    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.markdown("#### 输入图像")
        st.image(img, width="stretch")
    with img_col2:
        st.markdown("#### 边缘侧初筛结果")
        st.image(e_img, width="stretch")
    with img_col3:
        for r in targets: draw_academic_box(c_img, r['box'], f"{CLASSES[r['cls']]}", "#007bff")
        st.markdown("#### 云端协同分析结果")
        st.image(c_img, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 布局恢复：下层日志与指标 ---
    info_l, info_r = st.columns([2.5, 1.5])
    with info_l:
        st.markdown("#### 运行日志")
        st.markdown(f"<div class='academic-log-box'>{'<br>'.join(logs)}</div>", unsafe_allow_html=True)
    with info_r:
        st.markdown("#### 分析指标")
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("总时延", f"{t_total:.0f} ms" if t_total else "--")
        m2.metric("边缘时延", f"{t_e:.0f} ms" if t_e else "--")
        m3.metric("云端时延", (f"{t_c:.0f} ms" if cloud_hit else "0 ms (未上云)") if t_total else "--")
        m4.metric("目标数", len(targets) if t_total else "--")