# 硕士学位论文实验数据汇总
**课题**: 钢材表面缺陷检测的安全性与隐私保护研究
**数据集**: NEU-DET (目标域), Severstal (源域)
**模型**: YOLOv11n (Nano)

---

## 1. 实验环境与参数
* **GPU**: NVIDIA RTX 4090 (24GB)
* **Framework**: PyTorch 2.1 + Ultralytics 8.3
* **Image Size**: 640x640
* **Batch Size**: 16

## 2. 核心实验结果 (Performance Comparison)

| 实验 ID | 实验名称 | 策略描述 | mAP50 | 结论/意义 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | Baseline | ImageNet 预训练，无隐私保护 | **0.787** | 确立了任务的性能上限 (Upper Bound)。 |
| **Exp 2** | Pre-train | Severstal 域适应预训练 (30 Epochs) | 0.537 | 成功提取了钢材通用纹理特征 (Loss收敛)。 |
| **Exp 3** | Transfer | Severstal -> NEU-DET 迁移 (直接微调) | 0.748 | 出现负迁移 (Negative Transfer)，低于 Baseline。 |
| **Exp 4** | **SA-LDP (Ours)** | **结构感知层级差分隐私 (Epsilon=50)** | **0.692** | **核心成果**。在保护隐私的前提下，保留了 88% 的模型精度。 |

## 3. 安全性分析 (Attack Experiment)
采用梯度反转攻击 (DLG) 针对 Conv1 层进行重建。

| 攻击场景 | 防御策略 | 攻击迭代数 | 最终 MSE Loss | 视觉效果 | 结论 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | 无防御 (None) | 300 | **0.0047** | 轮廓清晰，纹理泄露 | 模型存在严重隐私风险，黑客可还原原图。 |
| **Scenario B** | **SA-LDP (Ours)** | 300 | **4.2673** | 纯噪声，无语义信息 | 攻击 Loss 高出 3 个数量级，防御有效。 |

## 4. 关键超参数 (Hyperparameters)
* **SA-LDP 配置**:
    * `Epsilon`: 50.0 (总预算)
    * `Delta`: 1e-5
    * `Clipping Threshold`: 10.0 (适应 YOLO 大梯度)
    * `Backbone Factor`: 0.5 (强噪)
    * `Head Factor`: 2.0 (弱噪)
* **训练配置**:
    * `Warmup`: 3 Epochs (前3轮不加噪)
    * `EMA`: Enabled (关键修复)