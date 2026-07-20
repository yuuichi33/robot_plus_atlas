# 机器人任务控制系统

> 硬件：Atlas 200I DK A2 + USB 摄像头 + 机器人底盘  
> 系统：Ubuntu 22.04 (aarch64), Python 3.9

## 分支说明

| 分支 | 用途 |
|------|------|
| `main` | 完整任务流程：立方体检测 → 绕行扫描文字 → 执行动作 → QR 验证 → 第二次动作 |
| `phase1` | 简化版：只做文字识别 + QR 识别 + 动作执行，无底盘移动 |

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
# 阶段一：文字识别 + 动作（底盘不动）
python3 phase1.py --camera 0 --port /dev/ttyUSB0

# 完整任务（底盘会移动）
python3 mission_main.py --vision --camera 0 --enable-chassis --port /dev/ttyUSB0 --display

# 纯视觉测试
python3 vision.py --mode block --camera 0
python3 vision.py --mode text --camera 0
```

### 开机自启

```bash
# phase1 服务
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

### 常见问题

| 问题 | 解决 |
|------|------|
| 摄像头打不开 | `ls /dev/video*` 确认编号，`modprobe -r uvcvideo; modprobe uvcvideo` 重载驱动 |
| 串口找不到 | 底盘上电了吗？`ls /dev/ttyUSB*` |
| 方块检测不到 | `python3 vision.py --mode block` 确认视野 |
| OCR 太慢 | ARM CPU 正常，已优化跳帧 + 缩放到 200px |
