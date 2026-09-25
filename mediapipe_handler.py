import mediapipe as mp
import cv2

class HandExtractor:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.6):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )

    def extract_hand_roi(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None

        h, w, _ = frame.shape
        hand = results.multi_hand_landmarks[0]
        x_min = min([lm.x for lm in hand.landmark]) * w
        x_max = max([lm.x for lm in hand.landmark]) * w
        y_min = min([lm.y for lm in hand.landmark]) * h
        y_max = max([lm.y for lm in hand.landmark]) * h

        # Expand the edges slightly
        margin = 20
        x_min = max(int(x_min - margin), 0)
        y_min = max(int(y_min - margin), 0)
        x_max = min(int(x_max + margin), w)
        y_max = min(int(y_max + margin), h)

        hand_crop = frame[y_min:y_max, x_min:x_max]
        return hand_crop
