# installer for Pi 400 + Pi 5 (USB + CSI)

#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="handpi"

APP_DIR="/home/pi/handpi_kiosk/pi"

VENV_DIR="${APP_DIR}/handpi"
ENTRYPOINT="${APP_DIR}/main_pi.py"
ROOT_ARG="${APP_DIR}"
IPC_SOCKET="/run/handpi/mpvsocket"

# prefer apt python3-opencv for ARM stability
PY_PKGS=(mediapipe)

echo "== HandPi Kiosk FINAL setup =="

if [[ "$(id -u)" -eq 0 ]]; then
  echo "ERROR: Do not run as root. Run as user 'pi' (sudo will be used when needed)."
  exit 1
fi

if [[ ! -d "${APP_DIR}" ]]; then
  echo "ERROR: APP_DIR does not exist: ${APP_DIR}"
  exit 1
fi

if [[ ! -f "${ENTRYPOINT}" ]]; then
  echo "ERROR: ENTRYPOINT not found: ${ENTRYPOINT}"
  exit 1
fi

if [[ ! -f "${APP_DIR}/mpv/scripts/automain.lua" ]]; then
  echo "ERROR: Missing Lua script: ${APP_DIR}/mpv/scripts/automain.lua"
  exit 1
fi

if [[ ! -d "${APP_DIR}/videos/main" ]]; then
  echo "ERROR: Missing videos/main folder: ${APP_DIR}/videos/main"
  exit 1
fi

echo "== Installing system packages (apt) =="
sudo apt update
sudo apt install -y \
  git \
  mpv \
  v4l-utils \
  libcamera-apps \
  python3-venv python3-pip \
  python3-opencv \
  python3-picamera2 python3-libcamera \
  python3-evdev \
  python3-numpy \
  build-essential \
  libglib2.0-0 libatlas-base-dev \
  fonts-noto-core fonts-noto-extra fonts-noto-color-emoji

echo "== Fix Windows CRLF (if needed) =="
dos2unix "${APP_DIR}/"*.sh >/dev/null 2>&1 || true
dos2unix "${ENTRYPOINT}" >/dev/null 2>&1 || true

echo "== Creating venv (system-site-packages) =="
if [[ -d "${VENV_DIR}" ]]; then
  echo "Venv exists: ${VENV_DIR}"
else
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

echo "== Ensuring venv sees system-site-packages =="
CFG="${VENV_DIR}/pyvenv.cfg"
if [[ -f "${CFG}" ]]; then
  if ! grep -q '^include-system-site-packages = true' "${CFG}"; then
    sudo sed -i 's/^include-system-site-packages.*/include-system-site-packages = true/; t; $a include-system-site-packages = true' "${CFG}"
  fi
fi

echo "== Upgrading pip tooling (venv) =="
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "== Installing Python packages into venv =="
"${VENV_DIR}/bin/python" -m pip install "${PY_PKGS[@]}"

echo "== Sanity checks (venv imports) =="
"${VENV_DIR}/bin/python" -c "import cv2; print('opencv OK')"
"${VENV_DIR}/bin/python" -c "import evdev; print('evdev OK')"
"${VENV_DIR}/bin/python" -c "from picamera2 import Picamera2; print('picamera2 OK')"

echo "== Ensuring runtime dirs exist =="
mkdir -p "${APP_DIR}/models"
mkdir -p "${APP_DIR}/videos/secondary"

echo "== TTY FIX: make tty1 kiosk-owned (stop login racing) =="
# Kiosk on tty1, login on tty2
sudo systemctl disable --now getty@tty1.service >/dev/null 2>&1 || true
sudo systemctl mask getty@tty1.service >/dev/null 2>&1 || true
sudo systemctl enable --now getty@tty2.service >/dev/null 2>&1 || true

echo "== Writing systemd service: /etc/systemd/system/${SERVICE_NAME}.service =="
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee "${SERVICE_PATH}" >/dev/null <<EOF
[Unit]
Description=HandPi Kiosk
After=network.target sound.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=${APP_DIR}

# Fresh runtime dir each boot, owned by service user
RuntimeDirectory=handpi
RuntimeDirectoryMode=0755

# Ensure stale socket never blocks startup
ExecStartPre=/bin/rm -f ${IPC_SOCKET}
ExecStartPre=/bin/sleep 2

# Force tty1 (kiosk console)
TTYPath=/dev/tty1
StandardInput=tty
StandardOutput=tty
StandardError=journal
TTYReset=yes
TTYVHangup=yes
TTYVTDisallocate=yes

# Hardware access
SupplementaryGroups=video input render

Environment=PYTHONUNBUFFERED=1
Environment=OPENCV_LOG_LEVEL=ERROR

ExecStart=${VENV_DIR}/bin/python ${ENTRYPOINT} --root ${ROOT_ARG} --ipc ${IPC_SOCKET}

Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

echo "== Reloading systemd + enabling service =="
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"

echo "== Reset-failed + starting service =="
sudo systemctl reset-failed "${SERVICE_NAME}.service" || true
sudo systemctl restart "${SERVICE_NAME}.service"

echo "== Status =="
systemctl --no-pager status "${SERVICE_NAME}.service" || true

echo
echo "✅ Installed."
echo "Logs: journalctl -u ${SERVICE_NAME}.service -f"
echo "Console login is now on tty2 (Ctrl+Alt+F2). Kiosk runs on tty1 (Ctrl+Alt+F1)."
