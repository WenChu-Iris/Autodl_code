import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import time

# ==========================================
# 0. 页面全局配置
# ==========================================
st.set_page_config(page_title="工业缺陷云边协同检测平台", page_icon="🏭", layout="wide")

# ==========================================
# 1. 模型结构定义 (核心依赖：必须放在最前面)
# ==========================================
# 全局模型参数
IMG_SIZE = 128
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 4

class DynamicAct(nn.Module):
    def __init__(self):
        super().__init__()
        # 固定系数 (HE 友好)
        self.register_buffer('a', torch.tensor(0.17))
        self.register_buffer('b', torch.tensor(0.5))
        self.register_buffer('c', torch.tensor(0.12))
    def forward(self, x): return self.a * (x**2) + self.b * x + self.c

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
        # Client 持有前 2 层
        self.blocks = nn.ModuleList([PolyBlock(EMBED_DIM, NUM_HEADS) for _ in range(2)])
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for blk in self.blocks: x = blk(x)
        return x

class ServerModel(nn.Module):
    def __init__(self, num_classes=6): # 注意这里默认改为微调后的 6 类
        super().__init__()
        # Server 持有后 2 层 + 分类头
        self.blocks = nn.ModuleList([PolyBlock(EMBED_DIM, NUM_HEADS) for _ in range(2)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, num_classes)
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


# 缺陷类别映射 (NEU-DET)
CLASSES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# ==========================================
# 2. 缓存加载模型 (避免每次点击按钮都重新加载大文件)
# ==========================================
@st.cache_resource
def load_system_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 【⚠️ 请确保这里的路径与你的 AutoDL 环境一致】
    YOLO_WEIGHTS = '/root/autodl-tmp/exp1/result_exp1/18_Rescue_Crazing/weights/best.pt'
    CLIENT_WEIGHTS = '/root/autodl-tmp/exp2/Fed_ViT_Models/fed_vit_client_best_v50.pth'
    SERVER_WEIGHTS = '/root/autodl-tmp/exp2/Fed_ViT_Models/fed_vit_server_best_v50.pth'
    
    # 加载 YOLO (边缘端初筛)
    yolo = YOLO(YOLO_WEIGHTS)
    
    # 加载 Client (边缘端特征提取)
    client = ClientModel().to(device)
    client.load_state_dict(torch.load(CLIENT_WEIGHTS, map_location=device))
    client.eval()
    
    # 加载 Server (云端复核)
    server = ServerModel(num_classes=6).to(device)
    server.load_state_dict(torch.load(SERVER_WEIGHTS, map_location=device))
    server.eval()
    
    return yolo, client, server, device

# 初始化系统模型
try:
    yolo_model, client_model, server_model, DEVICE = load_system_models()
    models_loaded = True
except Exception as e:
    st.error(f"模型加载失败，请检查权重路径！详细信息: {e}")
    models_loaded = False

# ViT 预处理工具
vit_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================
# 3. 界面交互与控制流
# ==========================================
st.title("🏭 基于隐私保护的云边协同缺陷检测平台")
st.markdown("演示：**边缘端 YOLO 快速初筛** 联合 **云端 Split-ViT 高精度复核**，并在传输中引入 DP 隐私噪声。")

# --- 侧边栏配置区 ---
with st.sidebar:
    st.header("⚙️ 协同策略配置")
    conf_thresh = st.slider(
        "边缘端独立决策阈值", 
        min_value=0.1, max_value=0.9, value=0.6, step=0.05,
        help="YOLO置信度高于此值时直接输出结果；低于此值则触发云端专家复核。"
    )
    noise_std = st.slider(
        "差分隐私传输噪声强度 (DP)", 
        min_value=0.0, max_value=0.01, value=0.001, step=0.001, format="%.4f",
        help="在边缘端上传特征到云端前注入的随机噪声，保障数据不被反推重建。"
    )
    st.divider()
    st.info(f"💻 当前算力引擎: **{DEVICE.upper()}**\n\n✅ 边缘端节点在线\n☁️ 云端计算节点在线")

# --- 主展示区 ---
col_img, col_log = st.columns([2, 1])

with col_img:
    st.subheader("🖼️ 缺陷图像上传")
    uploaded_file = st.file_uploader("请上传待检测的钢材表面图像 (JPG/PNG)", type=['jpg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert('RGB')
        display_image = original_image.copy()
        
        # 将按钮放在原图下方
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            st.image(original_image, caption="原始图像", use_container_width=True)
        with col_btn2:
            start_btn = st.button("🚀 开始检测", type="primary", use_container_width=True)
        
        # 当点击开始检测，并且模型加载成功时执行
        if start_btn and models_loaded:
            draw = ImageDraw.Draw(display_image)
            
            with col_log:
                st.subheader("📡 系统协同流转日志")
                log_container = st.container(height=450)
                
            # --- 【核心流水线】 ---
            with st.spinner("边缘端 YOLO 正在全图扫描..."):
                time.sleep(0.5) # 模拟系统启动延迟
                # 关键：YOLO conf 调到极低，迫使它找出所有可疑区域交给系统判断
                results = yolo_model(original_image, conf=0.05, iou=0.2, verbose=False)[0]
            
            stats = {'yolo_only': 0, 'cloud_called': 0}
            
            for box in results.boxes:
                conf = box.conf.item()
                cls_idx = int(box.cls.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # ------------------------------
                # 路线 A：边缘端独立解决 (置信度达标)
                # ------------------------------
                if conf >= conf_thresh:
                    stats['yolo_only'] += 1
                    label = f"{CLASSES[cls_idx]} (Edge:{conf:.2f})"
                    color = "#4CD964" # 绿色框
                    log_container.success(f"✅ [Edge] 检出 {CLASSES[cls_idx]} (把握: {conf:.2f})")
                
                # ------------------------------
                # 路线 B：呼叫云端复核 (置信度不足)
                # ------------------------------
                else:
                    stats['cloud_called'] += 1
                    log_container.warning(f"⚠️ [Edge] 发现疑似缺陷 (把握: {conf:.2f})，提取特征...")
                    
                    # 1. 边缘端：裁剪并提取中间特征 Z
                    crop_img = original_image.crop((x1, y1, x2, y2))
                    input_tensor = vit_transform(crop_img).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        z = client_model(input_tensor)
                        
                        # 2. 边缘端：注入隐私噪声并序列化传输
                        z_safe = z + torch.randn_like(z) * noise_std
                        log_container.info(f"🔒 [Net] 特征加密传输中 (DP Noise: {noise_std})...")
                        time.sleep(0.2) # 模拟网络延迟
                        
                        # 3. 云端：接收特征并输出高精度结果
                        logits = server_model(z_safe)
                        cloud_cls = logits.argmax(1).item()
                        cloud_conf = torch.softmax(logits, dim=1)[0][cloud_cls].item()
                        
                    label = f"{CLASSES[cloud_cls]} (Cloud:{cloud_conf:.2f})"
                    color = "#FF3B30" # 红色框
                    log_container.error(f"☁️ [Cloud] 复核完成！修正为 {CLASSES[cloud_cls]} ")
                
                # 在图像上绘制 BBox
                #draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                # 绘制文字标签底色
                #draw.rectangle([x1, y1-20, x1+160, y1], fill=color)
                #draw.text((x1+5, y1-15), label, fill="white")
                # 修改后：智能标签绘制
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2) # 边框变细一点

                # 计算文字大致长度 (根据字符数估算，每个字符约 6 像素宽)
                text_width = len(label) * 6 + 10 
                text_height = 15

                # 防止标签画到图片外面（如果框在最顶部，标签画在框内部）
                label_y1 = y1 - text_height if y1 - text_height > 0 else y1
                label_y2 = y1 if y1 - text_height > 0 else y1 + text_height

                # 绘制更紧凑的文字背景和文字
                draw.rectangle([x1, label_y1, x1 + text_width, label_y2], fill=color)
                draw.text((x1 + 3, label_y1 + 1), label, fill="white")
            
            st.divider()
            st.subheader("🎯 最终协同检测结果")
            st.image(display_image, caption="融合检测结果 (绿色: 边缘独立决策 | 红色: 云端协同复核)", use_container_width=True)
            
            # 展示统计指标看板
            m1, m2, m3 = st.columns(3)
            m1.metric(label="总发现异常区域", value=stats['yolo_only'] + stats['cloud_called'])
            m2.metric(label="边缘独立处理 (省带宽)", value=stats['yolo_only'])
            m3.metric(label="云端介入复核 (提精度)", value=stats['cloud_called'], delta="隐私传输保障")

    else:
        with col_log:
            st.info("👈 请先在左侧上传图像以启动工作流。")