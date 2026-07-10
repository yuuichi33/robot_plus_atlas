# Atlas 200I DK A2 机器人任务控制系统 — 部署与调试指南

> 硬件：Atlas 200I DK A2 + 1080P USB 摄像头 + 机器人底盘  
> 系统：Ubuntu 22.04 (aarch64), Python 3.9 (conda)  
> 工作目录：`~/robot_mission/260709/`  
> **当前设备**：摄像头 `/dev/video1`，底盘串口 `/dev/ttyUSB1`

---

## 一、部署

### 1. 安装依赖

```bash
pip3 install opencv-python numpy easyocr pyserial
```

### 2. 验证硬件

```bash
# 摄像头（每次重启后编号可能变，先用 ls 确认）
ls /dev/video*
python3 -c "import cv2; cap=cv2.VideoCapture(1); print(cap.read()[0])"

# 底盘串口（底盘需上电）
ls /dev/ttyUSB*
python3 -c "import serial; s=serial.Serial('/dev/ttyUSB1',115200); print('OK'); s.close()"
```

### 3. 摄像头打不开的排查

```bash
# 看 USB 总线上有没有摄像头
lsusb | grep -i cam

# 看内核日志有没有 USB 报错
dmesg | tail -20 | grep -i -E "usb|uvc|cam"

# 看 video 设备编号
ls /dev/video*

# 如果有设备但打不开，重载驱动
modprobe -r uvcvideo 2>/dev/null; modprobe uvcvideo; sleep 2; ls /dev/video*
```

---

## 二、调试命令

### 1. 纯视觉测试（不连串口）

```bash
python3 vision.py --mode block --camera 1
```

### 2. 底盘不动 + 动作测试

```bash
python3 mission_main.py --vision --camera 1 --port /dev/ttyUSB1 --display
```

底盘通电但不动，识别文字/QR 后会发串口执行砍刺动作。

### 3. 完整真实运行

```bash
python3 mission_main.py --vision --camera 1 --enable-chassis --port /dev/ttyUSB1 --display
```

### 4. 后台运行（断 SSH 不中断）

```bash
nohup python3 mission_main.py --vision --camera 1 --enable-chassis --port /dev/ttyUSB1 > /tmp/mission.log 2>&1 &
tail -f /tmp/mission.log
```

---

## 三、任务流程

```
SEARCH_BLOCK  转60° → 停5秒检测 → 找到则左右微调居中
    ↓
ORBIT_AND_SCAN  绕四方体移动，保持居中，扫描文字
    ├─ 四方体丢了 → 转60°/停5秒搜索找回
    ├─ 扫到文字 → 刹停 → 静止 OCR 确认（不限时）
    └─ 确认成功 → 执行第一次动作
    ↓
SEARCH_QR  原地转圈找 QR 码（不限时）
    ├─ 找到 → 靠近 → 验证 → 匹配则执行第二次动作
    └─ 继续找
    ↓
FINISHED
```

所有等待均为无限，只有成功才会推进。

---

## 四、命令行参数

| 参数 | 说明 |
|------|------|
| `--vision` | 启用真实摄像头 |
| `--camera 1` | 摄像头编号（当前是 1，每次重启确认） |
| `--port /dev/ttyUSB1` | 底盘串口（当前是 1） |
| `--enable-chassis` | 允许底盘移动 |
| `--dry-run` | 完全模拟，不连串口 |
| `--display` | 显示摄像头画面（按 q 退出） |
| `--no-qr` | 跳过 QR 扫描 |

---

## 五、开机自启

```bash
cat > /etc/systemd/system/robot-mission.service << 'EOF'
[Unit]
Description=Robot Mission Auto Start
After=multi-user.target

[Service]
Type=simple
WorkingDirectory=/root/robot_mission/260709
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/local/miniconda3/bin/python3 -u /root/robot_mission/260709/mission_main.py --vision --camera 1 --enable-chassis --port /dev/ttyUSB1
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
systemctl start robot-mission.service     # 启动
systemctl stop robot-mission.service      # 停止
systemctl disable robot-mission.service   # 取消自启
systemctl status robot-mission.service    # 查看状态
tail -100 /var/log/robot-mission.log      # 查看日志
> /var/log/robot-mission.log              # 清空日志
```

---

## 六、常见问题

| 问题 | 解决 |
|------|------|
| 摄像头打不开 | `ls /dev/video*` 确认编号，`modprobe -r uvcvideo; modprobe uvcvideo` 重载驱动 |
| 串口找不到 | 底盘上电了吗？`ls /dev/ttyUSB*` |
| 方块检测不到 | `python3 vision.py --mode block --camera 1` 确认视野 |
| OCR 太慢 | ARM CPU 正常，已优化跳帧+缩放到 200px |

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
