# 机器人任务控制系统

> 硬件：Atlas 200I DK A2 + USB 摄像头 + 智元素机器人  
> 系统：Ubuntu 22.04 (aarch64), Python 3.9  
> 仓库：https://github.com/yuuichi33/robot_plus_atlas

## 分支说明

| 分支 | 用途 |
|------|------|
| `main` | 完整任务流程（未完成）：立方体检测 → 绕行扫描文字 → 执行动作 → QR 验证 → 第二次动作 |
| `phase1` | 简化版：文字识别(EasyOCR) + QR 识别 + 动作执行，底盘由小程序 UDP 控制 |

> phase2 未完成，最后进度在 main，与 phase1 相比改进立方体识别算法，用 YOLOv5n 替代 EasyOCR，在 Atlas NPU 上推理。

## 模型说明（phase2 / YOLO）

`models/best.onnx` 是训练好的 YOLOv5n，5 分类：NoTask、Task1~4。

| Class | 含义 | 映射 |
|-------|------|------|
| Task1 | 位置1 劈砍 | pos=1, chop |
| Task2 | 位置2 劈砍 | pos=2, chop |
| Task3 | 位置1 刺击 | pos=1, stab |
| Task4 | 位置2 刺击 | pos=2, stab |


### ONNX → OM 转换（在 Atlas 上执行）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model=models/best.onnx \
    --framework=5 \
    --output=models/task_yolov5n_fp16 \
    --soc_version=Ascend310B4 \
    --input_shape="images:1,3,640,640" \
    --input_format=ND \
    --input_fp16_nodes="images" \
    --output_type=FP32
```

生成 `models/task_yolov5n_fp16.om`，由 `atlas_task/acl_backend.py` 加载在 NPU 上推理。

## 快速开始

### 安装依赖

```bash
pip3 install opencv-python numpy easyocr pyserial
```

### 确认设备

```bash
ls /dev/video*                      # 摄像头
ls /dev/ttyUSB* /dev/ttyACM*        # 底盘串口（需上电）
```

### 运行

```bash
# phase1：文字识别 + 动作（底盘不动）
python3 phase1.py --camera 0 --port /dev/ttyUSB0

# main：完整任务（底盘会移动）
python3 mission_main.py --vision --camera 0 --enable-chassis --port /dev/ttyUSB0 --display

# 纯视觉测试
python3 vision.py --mode block --camera 0
python3 vision.py --mode text --camera 0
```

### 开机自启

```bash
cat > /etc/systemd/system/phase1.service << 'EOF'
[Unit]
Description=Phase 1 Text Scan
After=multi-user.target
[Service]
Type=simple
WorkingDirectory=/root/robot_mission/260709
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -c 'PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -1); /usr/local/miniconda3/bin/python3 -u /root/robot_mission/260709/phase1.py --camera 0 --port ${PORT:-/dev/ttyUSB0}'
Restart=on-failure
RestartSec=10
User=root
StandardOutput=append:/var/log/phase1.log
StandardError=append:/var/log/phase1.log
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable phase1.service
```

