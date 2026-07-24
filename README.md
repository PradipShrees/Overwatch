# Overwatch

A small-scale face detection, tracking, and matching system for access control in
authorised settings — for example a licensed venue or a restricted facility where
only registered people may enter. Similar systems are already in commercial use by
major Australian retailers.

> **Responsible use:** face recognition is regulated in many jurisdictions
> (e.g. the Australian Privacy Act and biometric-data laws elsewhere). Only
> deploy this where you have clear legal authority and the informed consent of
> the people being recorded. Do a legal review before any real-world deployment.


Overwatch watches a Raspberry Pi camera feed, detects and tracks people, saves the
sharpest headshot of each person it sees, uploads it to Amazon S3, checks it against
a collection of known faces with AWS Rekognition, and sends an SNS alert when a
known face is matched. It runs fully headless — no monitor required — and keeps
working in local-only mode if AWS is unreachable.

## Features

- **Face detection** with OpenCV's YuNet model, run on a downscaled frame every
  few frames for good performance on a Pi
- **False-positive filtering** using size, aspect ratio, contrast, shadow ratio,
  edge density, and blur heuristics
- **Multi-person tracking** by bounding-box overlap (IoU) and centroid distance,
  with per-person movement trails
- **Re-identification** of people who briefly leave the frame, using HSV color
  histograms, with a configurable long-term memory (default: 30 minutes)
- **Best-shot capture** — keeps the sharpest crop per person and saves it the
  moment it passes a sharpness threshold (default: 200 Laplacian variance);
  people who never pass the bar get a fallback save when they leave
- **Cloud pipeline** — uploads headshots to S3, searches them against a
  Rekognition face collection, and publishes an SNS alert on a match
- **Graceful degradation** — with no AWS credentials, headshots are still saved
  and logged locally
- **Full logging** — rotating text log plus a CSV of every capture event
- **Live HUD** when a display is attached: bounding boxes, person IDs, sharpness
  scores, FPS, and a people-count sparkline

## Hardware

- Raspberry Pi 4 or 5 (64-bit Raspberry Pi OS, Bookworm or later)
- Raspberry Pi camera module (any model supported by Picamera2 / libcamera)

## Project layout

```
overwatch.py                        Main application
config.json                         All tunable settings
requirements.txt                    Python dependencies (pip)
setup.sh                            One-shot installer for a fresh Pi OS
face_detection_yunet_2023mar.onnx   YuNet model (downloaded by setup.sh)
headshots/                          Saved headshots (created at runtime)
overwatch.log                       Rotating application log
overwatch_events.csv                One row per saved headshot
```

## Installation

On a fresh Raspberry Pi OS, clone the repo and run the installer:

```bash
git clone https://github.com/PradipShrees/Overwatch.git
cd Overwatch
bash setup.sh
```

The installer:

1. Installs system packages with apt — `python3-picamera2` must come from apt
   because it is bound to the system libcamera and cannot be pip-installed
2. Creates a virtualenv at `~/overwatch-venv` with `--system-site-packages`
   so it can import the apt-installed Picamera2
3. Installs `opencv-python`, `numpy`, and `boto3` from `requirements.txt`
4. Downloads the YuNet face detection model from the
   [OpenCV Zoo](https://github.com/opencv/opencv_zoo) if it is missing
5. Verifies that every import works

## Configuration

All settings live in `config.json` next to the script:

```json
{
  "aws": {
    "region": "us-east-1",
    "s3_bucket": "headshotsforoverwatch",
    "s3_unknown_prefix": "unknown/",
    "rekognition_collection": "overwatch-known-faces",
    "match_threshold": 90,
    "sns_topic_arn": "arn:aws:sns:us-east-1:YOUR-ACCOUNT-ID:overwatch-alerts"
  },
  "camera": {
    "width": 1280,
    "height": 720,
    "target_fps": 15
  },
  "detection": {
    "detect_scale": 0.5,
    "detect_every_n_frames": 3,
    "yunet_score_threshold": 0.7,
    "yunet_nms_threshold": 0.3,
    "max_faces_per_frame": 10,
    "bbox_padding": 0.3,
    "min_face_size": 40,
    "min_contrast": 18,
    "min_dark_ratio": 0.05,
    "max_dark_ratio": 0.6,
    "min_edge_ratio": 0.03,
    "max_edge_ratio": 0.35,
    "min_laplacian_var": 60
  },
  "tracking": {
    "iou_threshold": 0.3,
    "max_centroid_distance": 120,
    "histogram_match_threshold": 0.35,
    "history_length": 30,
    "max_disappeared_frames": 45,
    "max_memory_frames": 450,
    "long_term_memory_minutes": 30,
    "min_save_sharpness": 200,
    "min_save_face_px": 80
  },
  "paths": {
    "log_file": "overwatch.log",
    "headshots_dir": "headshots",
    "yunet_model_path": "face_detection_yunet_2023mar.onnx"
  }
}
```

Key settings:

| Setting | Meaning |
| --- | --- |
| `tracking.min_save_sharpness` | Laplacian-variance sharpness a face must reach before its headshot is saved (200 = fairly strict) |
| `tracking.min_save_face_px` | Minimum face width/height in pixels for a quality save |
| `tracking.long_term_memory_minutes` | How long a person is remembered after leaving; returning within this window keeps the same ID and does not trigger a new headshot |
| `tracking.max_disappeared_frames` | Frames without a match before a track is considered gone (~3 s at 15 fps) |
| `detection.detect_every_n_frames` | Run the detector every Nth frame; lower = more responsive, more CPU |
| `aws.match_threshold` | Minimum Rekognition similarity (%) to count as a known-face match |

## AWS setup

Skip this section entirely if you only want local headshots — the app detects
missing credentials and keeps running in local-only mode.

Replace `REGION` (e.g. `us-east-1`) and `ACCOUNT_ID` (your 12-digit AWS account
ID) in everything below.

### 1. Create the S3 bucket

```bash
aws s3api create-bucket --bucket headshotsforoverwatch --region REGION \
  --create-bucket-configuration LocationConstraint=REGION
# omit --create-bucket-configuration if REGION is us-east-1

aws s3api put-public-access-block --bucket headshotsforoverwatch \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

### 2. Create the Rekognition collection

```bash
aws rekognition create-collection --collection-id overwatch-known-faces --region REGION
```

### 3. Index known faces

For each person you want recognized, upload one clear face photo and index it.
The `--external-image-id` is the name that appears in alerts.

```bash
aws s3 cp alice.jpg s3://headshotsforoverwatch/known/alice.jpg

aws rekognition index-faces \
  --collection-id overwatch-known-faces \
  --image '{"S3Object":{"Bucket":"headshotsforoverwatch","Name":"known/alice.jpg"}}' \
  --external-image-id alice \
  --max-faces 1 --region REGION
```

### 4. Create the SNS topic and subscribe

```bash
aws sns create-topic --name overwatch-alerts --region REGION
aws sns subscribe --topic-arn arn:aws:sns:REGION:ACCOUNT_ID:overwatch-alerts \
  --protocol email --notification-endpoint you@example.com
```

Confirm the subscription from the email you receive.

### 5. Create the IAM user

Save the following as `overwatch-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "UploadHeadshots",
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::headshotsforoverwatch/unknown/*"
    },
    {
      "Sid": "RekognitionReadsUploadedImage",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::headshotsforoverwatch/unknown/*"
    },
    {
      "Sid": "SearchFaces",
      "Effect": "Allow",
      "Action": "rekognition:SearchFacesByImage",
      "Resource": "arn:aws:rekognition:REGION:ACCOUNT_ID:collection/overwatch-known-faces"
    },
    {
      "Sid": "SendAlerts",
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:REGION:ACCOUNT_ID:overwatch-alerts"
    }
  ]
}
```

> `s3:GetObject` is required because Rekognition reads the uploaded image from
> S3 using the caller's permissions. The policy is runtime-only: steps 1–4 are
> one-time admin actions — run them with your own admin profile, not this user.

Then create the user and attach the policy:

```bash
aws iam create-user --user-name overwatch-pi
aws iam put-user-policy --user-name overwatch-pi \
  --policy-name overwatch-runtime --policy-document file://overwatch-policy.json
aws iam create-access-key --user-name overwatch-pi
```

The secret access key is shown only once — save it.

### 6. Configure credentials on the Pi

```bash
sudo apt-get install -y awscli
aws configure
```

Enter the access key, secret key, and `REGION`. Then set the same `REGION` and
your real `ACCOUNT_ID` in `config.json` (`aws.region` and `aws.sns_topic_arn`).

### 7. Verify

```bash
~/overwatch-venv/bin/python3 - <<'EOF'
import boto3, json
cfg = json.load(open('config.json'))['aws']
s3 = boto3.client('s3', region_name=cfg['region'])
s3.put_object(Bucket=cfg['s3_bucket'], Key='unknown/_test.jpg', Body=b'test')
print('S3 upload OK')
sns = boto3.client('sns', region_name=cfg['region'])
sns.publish(TopicArn=cfg['sns_topic_arn'], Subject='Overwatch test', Message='Setup verified.')
print('SNS publish OK')
EOF
```

## Usage

```bash
~/overwatch-venv/bin/python3 overwatch.py --headless
```

Without `--headless`, Overwatch auto-detects whether a display is present: with
one attached it shows the live HUD (press `q` to quit); without one it runs
headless (press `Ctrl+C` to quit).

### Run at boot (optional)

Create `/etc/systemd/system/overwatch.service`:

```ini
[Unit]
Description=Overwatch person tracker
After=network-online.target

[Service]
User=raspberry
WorkingDirectory=/home/raspberry/overwatch
ExecStart=/home/raspberry/overwatch-venv/bin/python3 overwatch.py --headless
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now overwatch
```

## Output

- **`headshots/person_<id>_<timestamp>.jpg`** — one headshot per tracked person
- **`overwatch.log`** — rotating application log (5 MB × 3 backups): startup,
  detections, saves, uploads, matches, alerts, errors
- **`overwatch_events.csv`** — one row per saved headshot:

```csv
timestamp,person_id,local_path,s3_key,upload_status
2026-07-05 23:00:12,1,headshots/person_1_20260705_230012.jpg,unknown/person_1_20260705_230012.jpg,uploaded
```

`upload_status` is `uploaded` or `local_only` (AWS unavailable or credentials
missing).

## How it works

1. A capture thread pulls frames from the camera at the target FPS.
2. Every Nth frame, YuNet detects faces on a downscaled copy; boxes are scaled
   back up, padded, and clipped.
3. Heuristic filters reject non-face detections (size, aspect, contrast,
   shadows, edges, blur).
4. Each valid face is matched to an existing track by IoU, then by centroid
   distance; unmatched faces are checked against recently disappeared people
   and long-term memory using HSV histogram similarity; otherwise a new person
   ID is created.
5. The sharpest crop per person is kept. When it passes the sharpness and size
   thresholds, a background thread saves it, uploads it to
   `s3://<bucket>/unknown/`, runs Rekognition `search_faces_by_image` against
   the known-faces collection, and publishes an SNS alert on a match.
6. Tracks expire in stages: active → disappeared (~30 s, cheap re-ID) →
   long-term memory (histogram only, default 30 min) → forgotten.

## Privacy and legal

This project captures and uploads images of people's faces and performs
biometric matching on them. Depending on where you live and where the camera
points, that can be regulated for example by the GDPR in the EU or biometric
privacy laws such as BIPA in Illinois and may require consent, notice, or
data-retention limits, especially if the camera sees visitors, neighbors, or
public space. Amazon Rekognition's own terms of service also apply. Use it
only where you have the right to record, and check your local rules before
deploying.

## License

[MIT](LICENSE) — free to use. If it helps you, a star is appreciated. :)
