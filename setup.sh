#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/overwatch-venv"
MODEL_FILE="${SCRIPT_DIR}/face_detection_yunet_2023mar.onnx"
MODEL_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "==> [1/5] Installing system packages (apt)"
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-picamera2 \
    libgl1 \
    libglib2.0-0 \
    libopenblas0 \
    curl

echo "==> [2/5] Creating virtualenv at ${VENV_DIR} (with system site packages)"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "==> [3/5] Installing Python packages (pip)"
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "==> [4/5] Fetching YuNet face detection model"
if [ ! -f "${MODEL_FILE}" ]; then
    curl -fsSL -o "${MODEL_FILE}" "${MODEL_URL}"
    echo "    downloaded $(basename "${MODEL_FILE}")"
else
    echo "    already present, skipping"
fi

echo "==> [5/5] Verifying imports"
python3 - <<'EOF'
import cv2, numpy, boto3, picamera2
assert hasattr(cv2, 'FaceDetectorYN'), "OpenCV build lacks FaceDetectorYN"
print(f"    cv2 {cv2.__version__} | numpy {numpy.__version__} | boto3 {boto3.__version__} | picamera2 OK")
EOF

echo
echo "Setup complete."
echo
echo "Run Overwatch with:"
echo "    ${VENV_DIR}/bin/python3 ${SCRIPT_DIR}/overwatch.py --headless"
echo
echo "Before S3 upload / Rekognition / SNS will work, configure AWS credentials:"
echo "    sudo apt-get install -y awscli   # optional, for 'aws configure'"
echo "    aws configure                    # or place keys in ~/.aws/credentials"
echo "Also edit config.json: set aws.region and your SNS topic ARN.If any error please ASK LLM"
