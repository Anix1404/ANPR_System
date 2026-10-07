import csv
import datetime
from collections import Counter, deque
import os
import random
import re
import time
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Force TCP transport and suppress verbose decoding logs
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|max_delay;500000"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

# -------------------------------------------------------------
# Configuration & Credentials Handling
# Priority: 1. Environment Variable -> 2. Local config file -> 3. Fallback prompt
# -------------------------------------------------------------
RTSP_URL = os.getenv("RTSP_URL")

if not RTSP_URL and os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as env_f:
        for line in env_f:
            if line.strip().startswith("RTSP_URL="):
                RTSP_URL = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                break

if not RTSP_URL:
    raise ValueError(
        "RTSP stream URL not set. Please export RTSP_URL as an environment variable "
        "or create a local .env file containing: RTSP_URL=rtsp://user:pass@ip:port/stream"
    )

# Storage Directories
BASE_DIR = os.getenv("ANPR_BASE_DIR", r"E:\ANPR_IIS")
VEHICLE_DIR = os.path.join(BASE_DIR, "snapshots", "vehicles")
PLATE_CROPS_DIR = os.path.join(BASE_DIR, "snapshots", "plates")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
CSV_FILE_PATH = os.path.join(BASE_DIR, "anpr_records.csv")

os.makedirs(VEHICLE_DIR, exist_ok=True)
os.makedirs(PLATE_CROPS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

COOLDOWN_SECONDS = 60
recent_logged_plates = {}
record_counter = 0

def init_csv(file_path):
    global record_counter
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["S.No", "Plate Number", "Quality Score", "Date", "Time", "Vehicle Snapshot", "Plate Crop", "Video Path"])
    else:
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            record_counter = max(0, len(reader) - 1)

def log_plate_to_csv(file_path, plate, score, vehicle_path, plate_crop_path, video_path):
    global record_counter
    record_counter += 1
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            record_counter, 
            plate, 
            f"{int(score)}", 
            date_str, 
            time_str, 
            vehicle_path or "N/A", 
            plate_crop_path or "N/A", 
            video_path or "N/A"
        ])
    print(f"[CSV LOGGED #{record_counter}] Plate: {plate} | Quality: {int(score)} | Time: {time_str}")

init_csv(CSV_FILE_PATH)

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en', enable_mkldnn=False)
print("PaddleOCR pipeline ready!")

# Load YOLO model
model_path = os.path.join(BASE_DIR, "best (1).pt")
if not os.path.exists(model_path):
    model_path = "best (1).pt"
yolo_model = YOLO(model_path)

STATE_CODES = (
    "AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|"
    "MH|ML|MN|MP|MZ|NL|OD|PB|PY|RJ|SK|TN|TR|TS|UK|UP|UT|WB"
)

INDIAN_PLATE_PATTERN = re.compile(
    rf"^(?:"
    rf"(?:{STATE_CODES})[0-9]{{1,2}}[A-Z]{{0,3}}[0-9]{{4}}|"
    rf"[0-9]{{2}}BH[0-9]{{4}}[A-Z]{{1,2}}|"
    rf"[0-9]{{1,3}}(?:CD|CC|UN)[0-9]{{1,4}}|"
    rf"[\^/A-Z]?[0-9]{{2}}[A-Z][0-9]{{5,6}}[A-Z]?"
    rf")$"
)

def validate_indian_plate(text):
    cleaned = re.sub(r'[^A-Z0-9\^]', '', text.upper())
    if cleaned.startswith("IND") and len(cleaned) > 5:
        cleaned = cleaned[3:]
    if INDIAN_PLATE_PATTERN.match(cleaned):
        return True, cleaned
    return False, cleaned

def is_frame_corrupted(crop):
    if crop is None or crop.size == 0:
        return True

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 20 or w < 50:
        return True

    patch_size = 16
    flat_patches = 0
    total_patches = 0
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            total_patches += 1
            if np.std(patch) < 3.0:
                flat_patches += 1

    if total_patches > 0 and (flat_patches / total_patches) > 0.40:
        return True

    rows_a = gray[7::8, :]
    rows_b = gray[8::8, :]
    min_len = min(len(rows_a), len(rows_b))

    if min_len > 0:
        diff_h = np.abs(rows_a[:min_len].astype(np.int16) - rows_b[:min_len].astype(np.int16))
        if np.mean(diff_h) > 28.0:
            return True

    return False

def compute_quality(crop):
    h, w = crop.shape[:2]
    if h < 25 or w < 100:
        return 0.0

    if is_frame_corrupted(crop):
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    raw_score = laplacian_var * (h * w)

    min_thresh = 300_000.0
    max_thresh = 15_000_000.0

    if raw_score < min_thresh:
        return 0.0

    normalized = 1.0 + ((raw_score - min_thresh) / (max_thresh - min_thresh)) * 99.0
    return float(np.clip(round(normalized, 1), 1.0, 100.0))

def parse_ocr_output(result):
    extracted = []
    if not result:
        return extracted
    for item in result:
        if isinstance(item, dict):
            res_dict = item.get("res", item)
            texts = res_dict.get("rec_texts", [res_dict.get("rec_text", "")])
            for text in texts:
                cleaned = re.sub(r'[^A-Z0-9\^]', '', str(text).upper())
                if cleaned and cleaned != "IND":
                    extracted.append(cleaned)
        elif isinstance(item, (list, tuple)):
            for line in item:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    entry = line[1]
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        text, score = str(entry[0]), float(entry[1])
                        if score >= 0.5:
                            cleaned = re.sub(r'[^A-Z0-9\^]', '', text.upper())
                            if cleaned and cleaned != "IND":
                                extracted.append(cleaned)
                    elif isinstance(entry, str):
                        cleaned = re.sub(r'[^A-Z0-9\^]', '', entry.upper())
                        if cleaned and cleaned != "IND":
                            extracted.append(cleaned)
    final_str = "".join(extracted)
    return [final_str] if final_str else []

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Failed to connect to RTSP stream.")
else:
    print("Successfully connected to RTSP stream.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0 or fps > 60:
    fps = 20.0

window_name = "RTSP Live Stream - ANPR"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

vehicle_tracker = {}
MAX_MISSED_FRAMES = 15
BUFFER_MAX_FRAMES = int(fps * 8)

while cap.isOpened():
    ret, img = cap.read()
    if not ret or img is None:
        print("Failed to grab frame or stream disconnected.")
        break

    h_img, w_img = img.shape[:2]
    annotated = img.copy()

    results = yolo_model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    current_active_ids = set()

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            current_active_ids.add(track_id)
            x1, y1, x2, y2 = box

            box_width = x2 - x1
            box_height = y2 - y1
            box_center_y = (y1 + y2) // 2

            if track_id not in vehicle_tracker:
                vehicle_tracker[track_id] = {
                    "best_score": 0.0,
                    "best_plate_crop": None,
                    "best_vehicle_frame": None,
                    "ocr_votes": [],
                    "frame_history": deque(maxlen=BUFFER_MAX_FRAMES),
                    "record_video": (random.random() < 0.35),
                    "missed_frames": 0,
                    "finalized": False
                }

            vehicle_tracker[track_id]["missed_frames"] = 0
            vehicle_tracker[track_id]["frame_history"].append(img.copy())

            is_at_toll_stop = (box_width >= 130) and (box_center_y >= int(h_img * 0.45))

            pw = int(box_width * 0.15)
            ph = int(box_height * 0.15)
            x1_pad = max(0, x1 - pw)
            y1_pad = max(0, y1 - ph)
            x2_pad = min(w_img, x2 + pw)
            y2_pad = min(h_img, y2 + ph)

            plate_crop = img[y1_pad:y2_pad, x1_pad:x2_pad]

            if is_at_toll_stop and plate_crop.size > 0 and not vehicle_tracker[track_id]["finalized"]:
                q_score = compute_quality(plate_crop)

                if q_score > vehicle_tracker[track_id]["best_score"]:
                    vehicle_tracker[track_id]["best_score"] = q_score
                    vehicle_tracker[track_id]["best_plate_crop"] = plate_crop.copy()
                    vehicle_tracker[track_id]["best_vehicle_frame"] = img.copy()

                cw = plate_crop.shape[1]
                ind_strip_width = int(cw * 0.12)
                ocr_crop = plate_crop[:, ind_strip_width:]

                if ocr_crop.size > 0 and q_score > 0.0:
                    try:
                        ocr_res = ocr(ocr_crop)
                    except Exception:
                        ocr_res = ocr.ocr(ocr_crop)

                    tokens = parse_ocr_output(ocr_res)
                    if tokens:
                        raw_candidate = tokens[0]
                        is_valid, cleaned_plate = validate_indian_plate(raw_candidate)
                        if is_valid:
                            vehicle_tracker[track_id]["ocr_votes"].append(cleaned_plate)

            display_text = "Scanning..."
            if vehicle_tracker[track_id]["ocr_votes"]:
                most_common = Counter(vehicle_tracker[track_id]["ocr_votes"]).most_common(1)
                display_text = most_common[0][0]

            color = (0, 255, 0) if is_at_toll_stop else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"ID:{track_id} | {display_text}", 
                        (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for t_id in list(vehicle_tracker.keys()):
        if t_id not in current_active_ids:
            vehicle_tracker[t_id]["missed_frames"] += 1
            if vehicle_tracker[t_id]["missed_frames"] >= MAX_MISSED_FRAMES:
                if not vehicle_tracker[t_id]["finalized"]:
                    votes = vehicle_tracker[t_id]["ocr_votes"]
                    best_score = vehicle_tracker[t_id]["best_score"]

                    if len(votes) >= 2 and best_score >= 40.0:
                        best_plate = Counter(votes).most_common(1)[0][0]
                        current_timestamp = time.time()

                        last_time = recent_logged_plates.get(best_plate, 0)
                        if (current_timestamp - last_time) >= COOLDOWN_SECONDS:
                            recent_logged_plates[best_plate] = current_timestamp
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                            vehicle_saved_path = None
                            if vehicle_tracker[t_id]["best_vehicle_frame"] is not None:
                                vehicle_saved_path = os.path.join(VEHICLE_DIR, f"{best_plate}_{timestamp_str}.jpg")
                                cv2.imwrite(vehicle_saved_path, vehicle_tracker[t_id]["best_vehicle_frame"])

                            plate_saved_path = None
                            if vehicle_tracker[t_id]["best_plate_crop"] is not None:
                                plate_saved_path = os.path.join(PLATE_CROPS_DIR, f"{best_plate}_{timestamp_str}.jpg")
                                cv2.imwrite(plate_saved_path, vehicle_tracker[t_id]["best_plate_crop"])

                            video_saved_path = None
                            frame_buffer = vehicle_tracker[t_id]["frame_history"]
                            min_duration_frames = int(fps * 5)

                            if vehicle_tracker[t_id]["record_video"] and len(frame_buffer) >= min_duration_frames:
                                video_saved_path = os.path.join(VIDEOS_DIR, f"{best_plate}_{timestamp_str}.mp4")
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                out_writer = cv2.VideoWriter(video_saved_path, fourcc, fps, (w_img, h_img))
                                for clip_frame in frame_buffer:
                                    out_writer.write(clip_frame)
                                out_writer.release()
                                print(f"[VIDEO ARCHIVED] Path: {video_saved_path}")

                            log_plate_to_csv(CSV_FILE_PATH, best_plate, best_score, vehicle_saved_path, plate_saved_path, video_saved_path)
                        else:
                            print(f"[COOLDOWN ACTIVE] Plate {best_plate} seen recently. Write skipped.")
                    else:
                        print(f"[IGNORED] Track ID {t_id}: Distant, corrupted, or insufficient valid votes.")

                    vehicle_tracker[t_id]["finalized"] = True
                del vehicle_tracker[t_id]

    display_w, display_h = 1280, 720
    display_img = cv2.resize(annotated, (display_w, display_h))
    cv2.imshow(window_name, display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()