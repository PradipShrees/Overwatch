#!/usr/bin/env python3

import os
import sys
import time
import math
import json
import csv
import threading
import logging
import logging.handlers
from datetime import datetime
from collections import deque
from pathlib import Path

HEADLESS = ('--headless' in sys.argv
            or not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))

if not HEADLESS:
    os.environ.setdefault('WAYLAND_DISPLAY', 'wayland-0')
    os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')

import cv2
import numpy as np
from picamera2 import Picamera2
import boto3
from botocore.exceptions import BotoCoreError, ClientError

CONFIG_PATH = Path(__file__).parent / 'config.json'
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

AWS   = CFG['aws']
CAM   = CFG['camera']
DET   = CFG['detection']
TRK   = CFG['tracking']
PATHS = CFG['paths']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            PATHS['log_file'], maxBytes=5*1024*1024, backupCount=3
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger('Overwatch')

Path(PATHS['headshots_dir']).mkdir(parents=True, exist_ok=True)

CSV_LOG = Path(PATHS['headshots_dir']).parent / 'overwatch_events.csv'
_csv_lock = threading.Lock()
if not CSV_LOG.exists():
    with open(CSV_LOG, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp', 'person_id', 'local_path', 's3_key', 'upload_status'])

def log_event(person_id, local_path, s3_key, upload_status):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with _csv_lock:
        with open(CSV_LOG, 'a', newline='') as f:
            csv.writer(f).writerow([ts, person_id, local_path, s3_key, upload_status])


try:
    s3_client  = boto3.client('s3',           region_name=AWS['region'])
    rek_client = boto3.client('rekognition',  region_name=AWS['region'])
    sns_client = boto3.client('sns',          region_name=AWS['region'])
    log.info("AWS clients initialized (S3, Rekognition, SNS)")
except Exception as e:
    log.warning(f"AWS init failed: {e}. Running in local-only mode.")
    s3_client = rek_client = sns_client = None


def upload_to_s3(local_path, person_id):
    if s3_client is None:
        return None
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_key = f"{AWS['s3_unknown_prefix']}person_{person_id}_{ts}.jpg"
    try:
        s3_client.upload_file(local_path, AWS['s3_bucket'], s3_key)
        log.info(f"  ↑ Uploaded to S3: {s3_key}")
        return s3_key
    except (BotoCoreError, ClientError) as e:
        log.error(f"  ✗ S3 upload failed for Person {person_id}: {e}")
        return None


def search_face(s3_key, person_id):
    if rek_client is None or s3_key is None:
        return
    try:
        resp = rek_client.search_faces_by_image(
            CollectionId=AWS['rekognition_collection'],
            Image={'S3Object': {'Bucket': AWS['s3_bucket'], 'Name': s3_key}},
            MaxFaces=1,
            FaceMatchThreshold=AWS['match_threshold'],
        )
        matches = resp.get('FaceMatches', [])
        if matches:
            match      = matches[0]
            similarity = match['Similarity']
            matched_id = match['Face']['ExternalImageId']
            log.info(f"  ★ MATCH: Person {person_id} → known/{matched_id} ({similarity:.1f}%)")
            if sns_client:
                sns_client.publish(
                    TopicArn=AWS['sns_topic_arn'],
                    Subject=f"Overwatch: Face Match Detected",
                    Message=(
                        f"Known face matched!\n\n"
                        f"Matched identity : {matched_id}\n"
                        f"Similarity       : {similarity:.1f}%\n"
                        f"Captured image   : s3://{AWS['s3_bucket']}/{s3_key}\n"
                        f"Time             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                )
                log.info(f"  ✉ SNS alert sent for match: {matched_id}")
        else:
            log.info(f"  – No match found for Person {person_id}")
    except (BotoCoreError, ClientError) as e:
        log.error(f"  ✗ Rekognition search failed for Person {person_id}: {e}")


class FrameCapture:
    def __init__(self, picam2):
        self._picam2 = picam2
        self._frame = None
        self._lock  = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        interval = 1.0 / CAM['target_fps']
        while self._running:
            t0 = time.time()
            frame = self._picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with self._lock:
                self._frame = frame
            elapsed = time.time() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False


def is_real_face(face_region):
    h, w = face_region.shape[:2]
    if h < DET['min_face_size'] or w < DET['min_face_size']:
        return False

    aspect = w / h if h > 0 else 0
    if not (0.6 <= aspect <= 1.4):
        return False

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    if gray.std() < DET['min_contrast']:
        return False

    mean_br = gray.mean()
    dark_ratio = np.sum(gray < (mean_br - 30)) / (w * h)
    if not (DET['min_dark_ratio'] <= dark_ratio <= DET['max_dark_ratio']):
        return False

    edges = cv2.Canny(gray, 40, 120)
    edge_ratio = cv2.countNonZero(edges) / (w * h)
    if not (DET['min_edge_ratio'] <= edge_ratio <= DET['max_edge_ratio']):
        return False

    if cv2.Laplacian(gray, cv2.CV_64F).var() < DET['min_laplacian_var'] * 0.5:
        return False

    return True


def sharpness(gray_face):
    return cv2.Laplacian(gray_face, cv2.CV_64F).var()


def face_histogram(bgr_face):
    hsv  = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def hist_distance(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1) * (iy2-iy1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def centroid_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def async_save_and_upload(person_id, best_frame_bgr):
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"person_{person_id}_{ts}.jpg"
    local = str(Path(PATHS['headshots_dir']) / fname)

    ok = cv2.imwrite(local, best_frame_bgr)
    if not ok:
        log.error(f"Failed to save {local}")
        return

    log.info(f"✓ Saved Person {person_id} → {fname}")
    s3_key = upload_to_s3(local, person_id)
    status = 'uploaded' if s3_key else 'local_only'
    log_event(person_id, local, s3_key or '', status)
    search_face(s3_key, person_id)


class PersonTracker:
    def __init__(self):
        self.persons     = {}
        self.disappeared = {}
        self.long_term   = {}
        self.next_id     = 1

    def _new_person(self, cx, cy, bbox, frame_bgr):
        pid = self.next_id
        self.next_id += 1
        gray_face = cv2.cvtColor(frame_bgr[bbox[1]:bbox[1]+bbox[3],
                                            bbox[0]:bbox[0]+bbox[2]],
                                  cv2.COLOR_BGR2GRAY)
        face_bgr  = frame_bgr[bbox[1]:bbox[1]+bbox[3],
                               bbox[0]:bbox[0]+bbox[2]].copy()
        self.persons[pid] = {
            'center':         (cx, cy),
            'bbox':           bbox,
            'history':        deque(maxlen=TRK['history_length']),
            'age':            0,
            'best_frame':     face_bgr,
            'best_sharpness': sharpness(gray_face),
            'histogram':      face_histogram(face_bgr),
            'first_seen':     datetime.now().strftime('%H:%M:%S'),
            'shot_saved':     False,
        }
        log.info(f"  + New person detected: P{pid}")
        return pid

    def update(self, faces, frame_bgr):
        valid_count  = 0
        matched_ids  = set()

        for (x, y, w, h) in faces:
            face_bgr = frame_bgr[y:y+h, x:x+w]
            if not is_real_face(face_bgr):
                continue

            valid_count += 1
            cx, cy = x + w//2, y + h//2
            bbox   = (x, y, w, h)

            best_id, best_iou = None, TRK['iou_threshold']
            for pid, pdata in self.persons.items():
                if pid in matched_ids:
                    continue
                iou = compute_iou(bbox, pdata['bbox'])
                if iou > best_iou:
                    best_iou, best_id = iou, pid

            if best_id is None:
                best_id, best_dist = None, TRK['max_centroid_distance']
                for pid, pdata in self.persons.items():
                    if pid in matched_ids:
                        continue
                    d = centroid_dist((cx, cy), pdata['center'])
                    if d < best_dist:
                        best_dist, best_id = d, pid

            if best_id is None:
                face_hist = face_histogram(face_bgr)
                for pid, pdata in list(self.disappeared.items()):
                    d_c = centroid_dist((cx, cy), pdata['center'])
                    d_h = hist_distance(face_hist, pdata['histogram'])
                    if d_c < 250 and d_h < TRK['histogram_match_threshold']:
                        best_id = pid
                        self.persons[pid] = pdata
                        del self.disappeared[pid]
                        log.info(f"  ↻ Re-identified Person {pid} (hist={d_h:.2f})")
                        break
            else:
                face_hist = None

            if best_id is None:
                if face_hist is None:
                    face_hist = face_histogram(face_bgr)
                for pid, ldata in list(self.long_term.items()):
                    if hist_distance(face_hist, ldata['histogram']) < TRK['histogram_match_threshold']:
                        gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                        self.persons[pid] = {
                            'center':         (cx, cy),
                            'bbox':           bbox,
                            'history':        deque(maxlen=TRK['history_length']),
                            'age':            0,
                            'best_frame':     face_bgr.copy(),
                            'best_sharpness': sharpness(gray_face),
                            'histogram':      face_hist,
                            'first_seen':     datetime.now().strftime('%H:%M:%S'),
                            'shot_saved':     ldata['shot_saved'],
                        }
                        del self.long_term[pid]
                        best_id = pid
                        log.info(f"  ↻ Long-term re-ID: P{pid} — shot_saved={ldata['shot_saved']}")
                        break

            if best_id is None:
                best_id = self._new_person(cx, cy, bbox, frame_bgr)
            else:
                self.persons[best_id]['center'] = (cx, cy)
                self.persons[best_id]['bbox']   = bbox
                self.persons[best_id]['age']    = 0

            gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            s = sharpness(gray_face)
            if s > self.persons[best_id]['best_sharpness']:
                self.persons[best_id]['best_sharpness'] = s
                self.persons[best_id]['best_frame']     = face_bgr.copy()
                self.persons[best_id]['histogram']      = face_hist if face_hist is not None else face_histogram(face_bgr)

            p = self.persons[best_id]
            if (not p['shot_saved']
                    and p['best_sharpness'] >= TRK['min_save_sharpness']
                    and w >= TRK['min_save_face_px']
                    and h >= TRK['min_save_face_px']):
                p['shot_saved'] = True
                log.info(f"  ★ Quality threshold met for P{best_id} "
                         f"(shp={p['best_sharpness']:.0f}, size={w}×{h}) — saving now")
                threading.Thread(
                    target=async_save_and_upload,
                    args=(best_id, p['best_frame'].copy()),
                    daemon=True
                ).start()

            self.persons[best_id]['history'].append((cx, cy))
            matched_ids.add(best_id)

        for pid in list(self.persons):
            self.persons[pid]['age'] += 1

        for pid in list(self.persons):
            if pid not in matched_ids and self.persons[pid]['age'] > TRK['max_disappeared_frames']:
                self.disappeared[pid] = self.persons.pop(pid)

        for pid in list(self.disappeared):
            self.disappeared[pid]['age'] += 1
            if self.disappeared[pid]['age'] >= TRK['max_memory_frames']:
                pdata = self.disappeared.pop(pid)
                shot_was_saved = pdata['shot_saved']
                if pdata['best_frame'] is not None and not shot_was_saved:
                    log.info(f"  ↓ Fallback save for P{pid} (never hit quality threshold)")
                    threading.Thread(
                        target=async_save_and_upload,
                        args=(pid, pdata['best_frame']),
                        daemon=True
                    ).start()
                    shot_was_saved = True
                self.long_term[pid] = {
                    'histogram':  pdata['histogram'],
                    'shot_saved': shot_was_saved,
                    'last_seen':  datetime.now(),
                }

        cutoff = datetime.now().timestamp() - TRK['long_term_memory_minutes'] * 60
        for pid in list(self.long_term):
            if self.long_term[pid]['last_seen'].timestamp() < cutoff:
                del self.long_term[pid]

        return self.persons, valid_count


COLORS = [
    (0, 255, 0), (255, 80, 0), (0, 165, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0)
]

_count_history = deque(maxlen=60)

def draw_hud(frame, tracked, fps, raw_faces, valid_faces):
    total  = len(tracked)
    reject = 0 if raw_faces == 0 else ((raw_faces - valid_faces) / raw_faces) * 100
    bar    = (f"FPS:{fps}  People:{total}  "
              f"Raw:{raw_faces}  Valid:{valid_faces}  Filter:{reject:.0f}%")
    cv2.putText(frame, bar, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

    cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

    for pid, pdata in tracked.items():
        color = COLORS[(pid - 1) % len(COLORS)]
        x, y, w, h = pdata['bbox']
        sharp = pdata['best_sharpness']

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"P{pid}  shp:{sharp:.0f}"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cx, cy = pdata['center']
        cv2.circle(frame, (cx, cy), 5, color, -1)

        pts = list(pdata['history'])
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, 1)

    _count_history.append(total)
    if len(_count_history) > 1:
        h_frame, w_frame = frame.shape[:2]
        sx, sy, sw, sh = w_frame - 130, 10, 120, 40
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (30, 30, 30), -1)
        max_c = max(_count_history) or 1
        pts_spark = []
        for i, c in enumerate(_count_history):
            px = sx + int(i * sw / len(_count_history))
            py = sy + sh - int(c / max_c * sh)
            pts_spark.append((px, py))
        for i in range(1, len(pts_spark)):
            cv2.line(frame, pts_spark[i-1], pts_spark[i], (0, 200, 255), 1)
        cv2.putText(frame, "people/60f", (sx+2, sy+sh+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)


def main():
    log.info("=" * 60)
    log.info("Overwatch — Starting up")
    log.info("=" * 60)

    picam2 = Picamera2()
    log.info(f"Sensor resolution: {picam2.sensor_resolution}")
    config = picam2.create_video_configuration(
        main={"format": 'BGR888', "size": (CAM['width'], CAM['height'])},
        raw={"size": picam2.sensor_resolution}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    capture = FrameCapture(picam2)

    scale = DET['detect_scale']
    det_w = int(CAM['width']  * scale)
    det_h = int(CAM['height'] * scale)
    face_detector = cv2.FaceDetectorYN.create(
        PATHS['yunet_model_path'],
        "",
        (det_w, det_h),
        score_threshold=DET['yunet_score_threshold'],
        nms_threshold=DET['yunet_nms_threshold'],
        top_k=DET['max_faces_per_frame']
    )

    tracker = PersonTracker()

    fps = frame_count = detect_frame_n = 0
    fps_time = time.time()
    last_faces = []

    if HEADLESS:
        log.info("Running HEADLESS (no preview window). Press Ctrl+C to quit.\n")
    else:
        log.info("Press 'q' to quit\n")

    try:
        while True:
            frame = capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            detect_frame_n += 1
            if detect_frame_n % DET['detect_every_n_frames'] == 0:
                small = cv2.resize(frame, (det_w, det_h))
                _, raw = face_detector.detect(small)
                if raw is not None and len(raw):
                    bboxes = raw[:, :4] / scale
                    h_frame, w_frame = frame.shape[:2]
                    pad = DET['bbox_padding']
                    clipped = []
                    for (bx, by, bw, bh) in bboxes:
                        pad_x = bw * pad * 0.5
                        pad_y = bh * pad * 0.5
                        bx -= pad_x
                        by -= pad_y
                        bw += pad_x * 2
                        bh += pad_y * 2
                        bx, by = max(0, int(bx)), max(0, int(by))
                        bw = min(int(bw), w_frame - bx)
                        bh = min(int(bh), h_frame - by)
                        if bw >= DET['min_face_size'] and bh >= DET['min_face_size']:
                            clipped.append((bx, by, bw, bh))
                    last_faces = clipped[:DET['max_faces_per_frame']]
                else:
                    last_faces = []

            tracked, valid = tracker.update(last_faces, frame)

            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()

            if not HEADLESS:
                draw_hud(frame, tracked, fps, len(last_faces), valid)
                cv2.imshow('Overwatch', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        capture.stop()
        picam2.close()
        cv2.destroyAllWindows()
        log.info("Overwatch stopped.")


if __name__ == '__main__':
    main()
