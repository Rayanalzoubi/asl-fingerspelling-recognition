# ============================================================
# 🎥 Extract hand images from a single video (local environment - Windows)
# ============================================================

import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# 🔹 Set the name of the video you want to extract from
video_path = "American Sign Language (ASL) fingerspelling.mp4"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"❌ لم يتم العثور على الفيديو: {video_path}")

# 🔹 Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.25  # low sensitivity to make detection easier
)
mp_draw = mp.solutions.drawing_utils

# 🔹 Create a folder to save the images
output_dir = "asl_single_finetune"
os.makedirs(output_dir, exist_ok=True)

# 🔹 Open the video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞️ جاري استخراج الإطارات من {video_path} ({total_frames} إطار)")

saved = 0
for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # Use only every 4th frame to reduce the count
    if frame_idx % 4 != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            h, w, _ = frame.shape
            coords = np.array([(lm.x * w, lm.y * h) for lm in hand.landmark])
            x_min, y_min = np.min(coords, axis=0).astype(int)
            x_max, y_max = np.max(coords, axis=0).astype(int)
            margin = 50
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)
            crop = frame[y_min:y_max, x_min:x_max]

            if crop.size > 0:
                filename = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
                cv2.imwrite(filename, crop)
                saved += 1

cap.release()
hands.close()

print(f"\n✅ تم استخراج {saved} صورة من الفيديو.")
print(f"📁 تم حفظها في المجلد: {output_dir}")
