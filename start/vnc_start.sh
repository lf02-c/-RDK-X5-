#!/bin/bash

# 设置必要的环境变量
export DISPLAY=:0
export XAUTHORITY=/home/elf/.Xauthority

# 可选：设置日志输出，便于调试
exec >> /home/elf/logs/vnc_5903.log 2>&1

echo "[$(date)] VNC 服务启动，端口 5903"

# 检查 X11 是否就绪
if ! xset -q > /dev/null 2>&1; then
    echo "错误：无法连接到 X11 显示服务器 $DISPLAY"
    exit 1
fi

# 启动 x11vnc 并前台运行（不要用 &）
x11vnc -rfbport 5903 -forever -shared -display $DISPLAY