import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw
import time
from datetime import datetime

# ==========================================
# 1. 核心架构：【GELU 版 Split-ViT】
# ==========================================
IMG_SIZE, PATCH_SIZE, EMBED_DIM = 128, 16, 256
CLASSES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

class ForwardDefense(nn.Module):
    """📍隐私加噪点：特征 Z 离开边缘侧前注入噪声"""
    def __init__(self):
        super().__init__()
        self.noise_std = 0.0 
    def forward(self, x):
        if self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

class StandardBlock(nn.Module):
    """📍架构更新：使用原生硬件加速的 GELU 激活函数"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(), # 🌟 核心替换：抛弃多项式，回归原生 GELU
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))

class ClientModel(nn.Module):
    """📍拆分点-边缘侧：持有前 2 层 Blocks"""
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
    """📍拆分点-云端侧：持有后 2 层 Blocks"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.blocks = nn.ModuleList([StandardBlock(EMBED_DIM, 4) for _ in range(2)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, num_classes)
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x).mean(dim=1))

# ==========================================
# 2. 算力适配：英伟达标准环境
# ==========================================
@st.cache_resource(show_spinner=False)
def load_system():
    # 检测英伟达算力卡
    device_cloud = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_edge = torch.device('cpu')
    
    # 路径配置
    Y_W = '/root/autodl-tmp/exp1/result_exp1/Final/weights/best.pt'
    C_W = '/root/autodl-tmp/exp3/Final_FL_Experiments_strict_pre10/best_client_weak_non_iid_with_defense.pth'
    S_W = '/root/autodl-tmp/exp3/Final_FL_Experiments_strict_pre10/best_server_weak_non_iid_with_defense.pth'
    
    yolo = YOLO(Y_W).to(device_edge)
    client = ClientModel().to(device_edge)
    client.load_state_dict(torch.load(C_W, map_location='cpu'))
    server = ServerModel().to(device_cloud)
    server.load_state_dict(torch.load(S_W, map_location=device_cloud))
    return yolo, client, server, device_edge, device_cloud

# 

# ==========================================
# 3. 业务逻辑与 UI 渲染
# ==========================================
st.set_page_config(page_title="端云协同-GPU版", layout="wide")
yolo_m, client_m, server_m, D_EDGE, D_CLOUD = load_system()
vit_tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def draw_academic_box(image, box, label, color):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    label_y = max(0, y1 - 14)
    draw.rectangle([x1, label_y, x1 + len(label)*7+8, label_y + 14], fill=color)
    draw.text((x1+3, label_y), label, fill="white")

# 顶部硬件看板
c1, c2, c3 = st.columns([4, 1.5, 1.5])
c1.markdown("### 端云协同缺陷分析原型系统 (GPU-Powered)")
c2.markdown(f"🖥️ **边缘端**: X86 CPU")
c3.markdown(f"☁️ **云端侧**: NVIDIA {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

with st.sidebar:
    st.header("实验配置")
    up_file = st.file_uploader("上传待检样本", type=['jpg','png','bmp'])
    thr = st.slider("协同判定阈值", 0.1, 0.9, 0.6)
    noise_val = st.slider("隐私加噪强度", 0.0, 0.05, 0.001, format="%.3f")
    run = st.button("执行协同推理", type="primary", use_container_width=True)

if up_file:
    img = Image.open(up_file).convert('RGB')
    e_img, c_img = img.copy(), img.copy()
    if run:
        t_start = time.time()
        y_res = yolo_m(img, conf=0.05, verbose=False)[0]
        results_list = []
        
        # 1. 边缘端初筛
        t0 = time.time()
        for b in y_res.boxes:
            cf, ci, box = b.conf.item(), int(b.cls.item()), b.xyxy[0].tolist()
            if cf >= thr:
                results_list.append({'box': box, 'cls': ci, 'source': 'Edge'})
                draw_academic_box(e_img, box, f"{CLASSES[ci]} {cf:.2f}", "#28a745")
            else:
                # 📍 加噪与特征提取
                client_m.forward_defense.noise_std = noise_val
                with torch.no_grad():
                    z = client_m(vit_tf(img.crop(box)).unsqueeze(0))
                results_list.append({'box': box, 'tensor': z, 'source': 'Cloud'})
                draw_academic_box(e_img, box, f"Uncertain {cf:.2f}", "#dc3545")
        t_e = (time.time() - t0) * 1000
        
        # 2. 云端侧协同
        t1 = time.time()
        cloud_hit = False
        for r in results_list:
            if r['source'] == 'Cloud':
                cloud_hit = True
                with torch.no_grad():
                    r['cls'] = torch.argmax(server_m(r['tensor'].to(D_CLOUD))).item()
        t_c = (time.time() - t1) * 1000 if cloud_hit else 0
        
        # 3. UI 结果反馈
        i1, i2, i3 = st.columns(3)
        i1.image(img, caption="原始输入图像")
        i2.image(e_img, caption="边缘侧初筛结果 (鲲鹏)")
        for r in results_list: draw_academic_box(c_img, r['box'], f"{CLASSES[r['cls']]}", "#007bff")
        i3.image(c_img, caption="云端协同最终判定 (GPU)")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("总处理时延", f"{(time.time()-t_start)*1000:.0f} ms")
        m2.metric("边缘计算耗时", f"{t_e:.0f} ms")
        m3.metric("云端加速耗时", f"{t_c:.0f} ms" if cloud_hit else "0 ms (本地闭环)")