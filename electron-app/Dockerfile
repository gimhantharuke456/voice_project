FROM node:18-bookworm

# Electron is a native GUI app - it needs a display. This image runs it
# inside a virtual X server (Xvfb) with a VNC server, exposed to the
# outside world over noVNC so you can view/interact with it from a browser.
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb fluxbox x11vnc novnc websockify \
    libgtk-3-0 libnotify4 libnss3 libxss1 libasound2 libxtst6 \
    xdg-utils libatspi2.0-0 libuuid1 libsecret-1-0 libgbm1 libdrm2 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user - Electron/Chromium and the VNC stack don't need
# root, and this limits the blast radius if the exposed VNC/websockify
# surface is ever compromised.
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser

ENV DISPLAY=:99

# noVNC web UI
EXPOSE 6080

ENTRYPOINT ["docker-entrypoint.sh"]
