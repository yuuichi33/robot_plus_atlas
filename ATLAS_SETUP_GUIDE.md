# Atlas 200I DK A2 机器人任务控制系统 — 部署指南

> 硬件：Atlas 200I DK A2 (Ascend 310B4, CANN 7.0) + Hikvision USB 摄像头  
> 连接：PC <--USB--> Atlas (虚拟网卡, SSH)  
> 系统：Ubuntu 22.04 (aarch64), Python 3.9 (conda)

---

## 目录

1. [连接 Atlas](#1-连接-atlas)
2. [上传代码](#2-上传代码)
3. [安装依赖](#3-安装依赖)
4. [验证硬件](#4-验证硬件)
5. [逐步测试](#5-逐步测试)
6. [任务流程](#6-任务流程)
7. [命令行参数](#7-命令行参数)
8. [常见问题](#8-常见问题)


### 4.3 验证安装

```bash
python3 -c "import cv2; print('OpenCV', cv2.__version__)"
python3 -c "import serial; print('pyserial OK')"
python3 -c "import numpy; print('numpy', numpy.__version__)"

# 验证 OCR（如果装了 PaddleOCR）
python3 -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
```

---

## 5. 验证硬件

### 5.1 验证摄像头

```bash
# 在 Atlas 上执行
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print(f'摄像头正常，画面尺寸: {frame.shape}')
else:
    print('无法读取摄像头画面')
cap.release()
"
```

如果报错，尝试：
```bash
# 检查摄像头权限
ls -la /dev/video0

# 如果是权限问题，把自己加入 video 组
sudo usermod -a -G video $USER
# 然后重新登录
```

### 5.2 验证串口（机器人底盘通信）

```bash
# 查看可用串口
ls /dev/ttyUSB* /dev/ttyACM*

# 如果有 /dev/ttyUSB0，测试通信
python3 -c "
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
print('串口打开成功')
ser.close()
"
```

如果串口权限不足：
```bash
sudo usermod -a -G dialout $USER
# 重新登录后生效
```

### 5.3 单独测试串口通信

```bash
cd ~/robot_mission

# 只测试 PING 和 STOP（不会让机器人移动）
python3 bridge_usb_test.py
# 按 Enter 逐步执行，观察机器人是否有响应
```

---

## 6. 逐步调试运行

> ⚠️ **安全警告**：测试动作前确保机器人周围有足够空间，人和障碍物远离机器人手臂！

### 第 1 步：纯软件模拟（确认逻辑正确）

```bash
cd ~/robot_mission

# 完全 dry-run，不连接任何硬件
python3 mission_main.py --dry-run
```

期望输出：
```
[INIT] 🚀 任务启动
[SEARCH_BLOCK] 🔍 开始搜索纯色立方体柱子...
[SEARCH_BLOCK] step 00: found=False, area=0
[SEARCH_BLOCK] step 01: found=False, area=0
[SEARCH_BLOCK] step 02: found=True, area=12000
[SEARCH_BLOCK] 🎯 检测到方块！
[APPROACH_BLOCK] 🚶 靠近方块...
[CIRCLE_SCAN_TEXT] 🔄 绕柱子扫描 A4 纸文字...
[CIRCLE_SCAN_TEXT] 📝 识别成功！位置1 → chop
[EXECUTE_TASK] ⚔️ 执行动作: 位置1 → chop
[FINISHED] 🏁 任务完成！
```

### 第 2 步：测试摄像头 + 模拟底盘

```bash
# 使用真实摄像头检测，但不动底盘
python3 mission_main.py --vision --camera 0 --mock-position 1 --mock-attack chop
```

这时摄像头画面会实时处理：
- 把方块放在摄像头前，应该能看到 `🎯 检测到方块！`
- 把写有 "位置1 劈砍" 的 A4 纸放在摄像头前，应该能看到 `📝 识别成功！`

### 第 3 步：单独测试视觉模块

```bash
# 只测试方块检测
python3 vision.py --mode block --camera 0

# 只测试 A4 纸 OCR 文字识别
python3 vision.py --mode text --camera 0

# 只测试二维码扫描
python3 vision.py --mode qr --camera 0

# 颜色标定工具（点击画面中的颜色来获取 HSV 范围）
python3 vision.py --mode calibrate --camera 0
```

### 第 4 步：测试机器人单个动作

```bash
# 测试通信 + 单个动作（需要按 Enter 确认安全）
python3 test_robot_control.py --port /dev/ttyUSB0 --action 0
# ACTION_0 = 位置1劈砍，ACTION_1 = 位置2劈砍
# ACTION_2 = 位置1刺击，ACTION_3 = 位置2刺击
```

### 第 5 步：连接底盘 + 真实视觉（完整测试）

```bash
# ⚠️ 完整运行，机器人会真实移动和做动作！
# 建议先用 --no-qr 跳过二维码扫描
python3 mission_main.py \
    --vision --camera 0 \
    --enable-chassis \
    --port /dev/ttyUSB0 \
    --no-qr
```

---

## 7. 完整任务运行

确认所有模块正常后，完整流程：

```bash
cd ~/robot_mission

python3 mission_main.py \
    --vision \
    --camera 0 \
    --enable-chassis \
    --enable-qr \
    --port /dev/ttyUSB0 \
    --block-colors red blue green yellow
```

### 命令行参数速查

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--port` | Atlas 串口设备路径 | `/dev/ttyUSB0` |
| `--baudrate` | 串口波特率 | `115200` |
| `--vision` | 启用真实摄像头视觉 | 关闭（用桩模块） |
| `--camera` | 摄像头设备 ID | `0` |
| `--block-colors` | 要检测的方块颜色列表 | `red blue green yellow purple orange` |
| `--enable-chassis` | 启用真实底盘移动 | 关闭 |
| `--enable-qr` | 启用二维码验证 | 开启 |
| `--no-qr` | 禁用二维码验证 | — |
| `--dry-run` | 完全模拟模式 | 关闭 |
| `--save-debug-frames` | 保存视觉调试帧 | 关闭 |
| `--debug-dir` | 调试帧保存目录 | `/tmp/robot_debug` |

---

## 8. 常见问题

### Q1: `Permission denied: /dev/video0`

```bash
sudo usermod -a -G video $USER
# 退出 SSH 重新登录
```

### Q2: `Permission denied: /dev/ttyUSB0`

```bash
sudo usermod -a -G dialout $USER
# 退出 SSH 重新登录
```

### Q3: OpenCV 报 `libGL.so.1: cannot open`

```bash
sudo apt install -y libgl1-mesa-glx
```

### Q4: PaddleOCR 下载模型太慢 / 失败

```bash
# 设置代理（如果有）
export HTTP_PROXY=http://你的代理IP:端口
export HTTPS_PROXY=http://你的代理IP:端口

# 或者改用 EasyOCR
pip3 uninstall paddlepaddle paddleocr -y
pip3 install easyocr
```

### Q5: 摄像头画面全黑

```bash
# 检查摄像头是否被其他进程占用
sudo fuser /dev/video0

# 杀死占用进程
sudo fuser -k /dev/video0
```

### Q6: 方块检测不到

1. 先用颜色标定工具获取准确的 HSV 范围：
   ```bash
   python3 vision.py --mode calibrate --camera 0
   ```
2. 点击画面中方块的颜色区域，记下输出的 HSV 值
3. 修改 `vision.py` 中 `_COLOR_RANGES` 字典对应的颜色范围

### Q7: OCR 识别不到中文

1. 确保安装了 PaddleOCR（中文效果最好）：
   ```bash
   pip3 install paddlepaddle paddleocr
   ```
2. 确保文字清晰、光照充足
3. A4 纸不要反光
4. 可以增大字体（打印大号字）

### Q8: 如何查看机器人串口返回的原始数据？

```bash
# 修改 bridge_usb_test.py，或者直接用这个命令监控串口
python3 -c "
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
while True:
    line = ser.readline()
    if line:
        print('RX:', line.decode(errors='replace').strip())
"
```

---

## 📁 项目文件结构

```
~/robot_mission/           # Atlas 上的工作目录
├── robot_control.py       # 机器人底层控制（串口通信）
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
