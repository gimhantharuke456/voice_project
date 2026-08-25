#!/bin/bash
set -e

# Virtual display
Xvfb :99 -screen 0 1280x900x24 &
sleep 1

# Lightweight window manager so the Electron window renders properly
fluxbox &
sleep 1

# VNC server on the virtual display
x11vnc -display :99 -forever -shared -nopw -rfbport 5900 -bg -o /var/log/x11vnc.log

# noVNC web client - proxies browser websocket -> VNC port 5900
websockify --web=/usr/share/novnc 6080 localhost:5900 &

# Launch Electron. --no-sandbox is required since we're running as root
# in the container (Chromium's sandbox refuses to init as root otherwise).
exec npx electron --no-sandbox --disable-gpu .
