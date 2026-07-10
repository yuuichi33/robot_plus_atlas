# Atlas 200I DK A2 机器人任务控制系统 — 部署与调试指南

> 硬件：Atlas 200I DK A2 + Hikvision USB 摄像头 + 机器人底盘  
> 系统：Ubuntu 22.04 (aarch64), Python 3.9 (conda)  
> 工作目录：`~/robot_mission/260709/`

---

## 一、部署

### 1. 安装依赖

```bash
pip3 install opencv-python numpy easyocr pyserial
```

### 2. 验证硬件

```bash
# 摄像头
ls /dev/video*
python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.read()[0])"

# 底盘串口（底盘需上电）
ls /dev/ttyUSB*
python3 -c "import serial; s=serial.Serial('/dev/ttyUSB0',115200); print('OK'); s.close()"
```

---

## 二、调试命令

### 1. 纯视觉测试（底盘不动，不连串口）

```bash
# 方块检测
python3 vision.py --mode block

# 文字识别
python3 vision.py --mode text
```

### 2. 视觉 + 动作测试（底盘上电、不动）

```bash
python3 mission_main.py --vision --camera 0 --port /dev/ttyUSB0 --display
```

底盘通电但不动，识别到文字和 QR 后会发串口指令执行动作。

### 3. 完整真实运行（底盘会上电移动）

```bash
python3 mission_main.py --vision --camera 0 --enable-chassis --port /dev/ttyUSB0 --display
```

机器人会转圈搜索 → 绕行扫描文字 → 识别后执行动作 → 搜索 QR → 验证后执行动作。

---

## 三、命令行参数

| 参数 | 说明 |
|------|------|
| `--vision` | 启用真实摄像头 |
| `--camera 0` | 摄像头编号 |
| `--port /dev/ttyUSB0` | 底盘串口 |
| `--enable-chassis` | 允许底盘移动（不加则只发动作不移动） |
| `--dry-run` | 完全模拟，不连串口 |
| `--display` | 显示摄像头实时画面（按 q 退出） |
| `--no-qr` | 跳过 QR 扫描阶段 |

---

## 四、开机自启

```bash
cat > /etc/systemd/system/robot-mission.service << 'EOF'
[Unit]
Description=Robot Mission Auto Start
After=multi-user.target

[Service]
Type=simple
WorkingDirectory=/root/robot_mission/260709
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/local/miniconda3/bin/python3 -u /root/robot_mission/260709/mission_main.py --vision --camera 0 --enable-chassis --port /dev/ttyUSB0
Restart=no
User=root
StandardOutput=append:/var/log/robot-mission.log
StandardError=append:/var/log/robot-mission.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable robot-mission.service
```

### 管理命令

```bash
systemctl start robot-mission.service    # 手动启动
systemctl stop robot-mission.service     # 停止
systemctl disable robot-mission.service  # 取消自启
tail -100 /var/log/robot-mission.log     # 查看日志
```

---

## 五、常见问题

### 摄像头打不开

```bash
ls /dev/video*                           # 确认设备存在
sudo fuser -k /dev/video0                # 杀掉占用进程
```

### 串口找不到

底盘必须上电。检查：
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

### 方块检测不到

用摄像头窗口模式确认视野：
```bash
python3 vision.py --mode block
```

### OCR 太慢

已优化：跳帧 + ROI 缩放到 200px。可在 `vision.py` 中调整 `_ocr_skip_interval`（默认 5）和 `_ocr_max_width`（默认 200）。

### 日志为空 / 服务挂了

```bash
tail -50 /var/log/robot-mission.log
systemctl status robot-mission.service
```
├── mission_main.py        # 任务状态机（主程序入口）
├── vision.py              # 视觉感知（OpenCV + OCR + QR）
├── bridge_usb_test.py     # 原始串口测试工具
├── test_robot_control.py  # 单指令测试工具
└── requirements.txt       # Python 依赖清单
```

## 🚀 快速启动清单

- [ ] Atlas SSH 可登录
- [ ] 摄像头 `/dev/video0` 可读取
- [ ] 串口 `/dev/ttyUSB0` 可打开
- [ ] Python 依赖全部安装成功
- [ ] `--dry-run` 模式跑通
- [ ] `--vision` 模式方块检测正常
- [ ] `--vision` 模式 OCR 识别正常
- [ ] `bridge_usb_test.py` 通信测试通过
- [ ] `--enable-chassis` 底盘移动测试通过
- [ ] 完整任务流程跑通 🎉
