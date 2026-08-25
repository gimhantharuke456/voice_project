#!/bin/bash
set -e

if [ -z "$VNC_PASSWORD" ]; then
    echo "ERROR: VNC_PASSWORD is not set." >&2
    echo "Run the container with -e VNC_PASSWORD=<something> to set a password" >&2
    echo "for the remote desktop session. Refusing to start with no auth." >&2
    exit 1
fi

# Virtual display
Xvfb :99 -screen 0 1280x900x24 &
sleep 1

# Lightweight window manager so the Electron window renders properly
fluxbox &
sleep 1

# VNC server on the virtual display, password-protected
x11vnc -display :99 -forever -shared -passwd "$VNC_PASSWORD" -rfbport 5900 -bg -o /tmp/x11vnc.log

# noVNC web client - proxies browser websocket -> VNC port 5900
# (the browser is prompted for the VNC password on connect)
websockify --web=/usr/share/novnc 6080 localhost:5900 &

# Launch Electron. --no-sandbox is required since we're running as root
# in the container (Chromium's sandbox refuses to init as root otherwise).
exec npx electron --no-sandbox --disable-gpu .
