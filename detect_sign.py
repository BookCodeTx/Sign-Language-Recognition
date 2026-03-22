import cv2
import mediapipe as mp  # pyright: ignore[reportMissingImports]
import numpy as np
import pickle
import os
import time
import threading
from gtts import gTTS  # pyright: ignore[reportMissingImports]
import pygame  # pyright: ignore[reportMissingImports]
from PIL import Image, ImageFont, ImageDraw  # pyright: ignore[reportMissingImports]


class SignLanguageDetector:
    def __init__(self, model_path="models/sign_language_model.pkl"):
        # 1. Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # ใช้มือเดียวเท่านั้น (ตัดฟีเจอร์ two hand ออก)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # 2. Load Models
        self.model = None
        self.labels_map = None
        self.load_model(model_path)

        # 3. ตั้งค่าภาษาไทยและการแสดงผล
        self.font_path = "C:/Windows/Fonts/tahoma.ttf" # สำหรับ Windows หรือแก้เป็น path ฟอนต์ของคุณ
        self.thai_labels = {
            "HELLO": "สวัสดี",
            # เพิ่มคำแปลที่นี่...
        }

        self.last_spoken_text = ""
        self.last_spoken_time = 0.0
        self.is_speaking = False
        pygame.mixer.init()

        # 4. History สำหรับ Smoothing
        self.prediction_history_left = []
        self.prediction_history_right = []
        self.history_size = 5

    def _run_speech(self, text):
        """ระบบเสียง Google พูดภาษาไทยแบบช้า"""
        self.is_speaking = True
        temp_file = "speech_temp.mp3"
        try:
            # slow=True จะทำให้ AI พูดช้าลงตามที่ต้องการ
            tts = gTTS(text=text, lang='th', slow=True) 
            tts.save(temp_file)
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
        finally:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except: pass
            self.is_speaking = False

    def speak_text(self, text):
        if not text or self.is_speaking:
            return
        
        thai_text = self.thai_labels.get(text, text)
        current_time = time.time()
        
        # เปลี่ยนท่าพูดทันที / ท่าเดิมรอ 3 วินาที
        if thai_text != self.last_spoken_text or (current_time - self.last_spoken_time > 3.0):
            self.last_spoken_text = thai_text
            self.last_spoken_time = current_time
            threading.Thread(target=self._run_speech, args=(thai_text,), daemon=True).start()

    def draw_thai_text(self, img, text, position, font_size, color):
        """ฟังก์ชันวาดภาษาไทยลงบนเฟรม (แก้ปัญหา ???? )"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.labels_map = model_data["labels_map"]

    def normalize_landmarks(self, landmarks):
        coords = np.array([[l.x, l.y, l.z] for l in landmarks])
        coords = coords - coords[0]
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        return (coords / max_dist).flatten() if max_dist > 0 else coords.flatten()

    def get_prediction(self, landmarks):
        feat = self.normalize_landmarks(landmarks).reshape(1, -1)
        pred = self.model.predict(feat)[0]
        conf = self.model.predict_proba(feat)[0][pred]
        return self.labels_map[pred], conf

    def smooth_prediction(self, prediction, hand_type="left"):
        history = self.prediction_history_left if hand_type == "left" else self.prediction_history_right
        history.append(prediction)
        if len(history) > self.history_size: history.pop(0)
        return max(set(history), key=history.count) if len(history) >= 3 else prediction

    def detect_realtime(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            lp, lc, rp, rc = None, 0.0, None, 0.0

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                h_list = [h.landmark for h in results.multi_hand_landmarks]
                try:
                    for i, lms in enumerate(h_list):
                        p, c = self.get_prediction(lms)
                        hand_type = "left" if i == 0 else "right"
                        smoothed = self.smooth_prediction(p, hand_type)
                        
                        if hand_type == "left":
                            lp, lc = smoothed, c
                        else:
                            rp, rc = smoothed, c
                except:
                    pass
            else:
                self.prediction_history_left = []
                self.prediction_history_right = []
                self.last_spoken_text = ""

            # วาด UI และคืนค่าเฟรมที่แปลงแล้ว
            frame = self.draw_ui(frame, lp, lc, rp, rc)
            
            cv2.imshow("Sign Language Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

        cap.release()
        cv2.destroyAllWindows()

    def draw_ui(self, frame, lp, lc, rp, rc):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        final_label = None
        final_conf = 0.0

        # แสดงข้อความซ้าย/ขวา
        if lp:
            frame = self.draw_thai_text(frame, f"LEFT: {self.thai_labels.get(lp, lp)}", (20, 50), 30, (0, 255, 255))
            final_label, final_conf = lp, lc
        if rp:
            frame = self.draw_thai_text(frame, f"RIGHT: {self.thai_labels.get(rp, rp)}", (20, 100), 30, (255, 165, 0))
            if not final_label or rc > lc: 
                final_label, final_conf = rp, rc

        # แสดงตัวอักษรใหญ่ที่คำนวณตำแหน่งอัตโนมัติ (ไม่โดนตัด)
        if final_label:
            self.speak_text(final_label)
            display_text = self.thai_labels.get(final_label, final_label)
        
            # คำนวณความกว้างตัวอักษรเพื่อจัดตำแหน่ง (ประมาณ 50px ต่อตัว)
            text_width = len(display_text) * 50
            tx = max(10, w - text_width - 50)
        
            color = (0, 255, 0) if final_conf > 0.8 else (0, 255, 255)
            frame = self.draw_thai_text(frame, display_text, (tx, 50), 90, color)
            cv2.putText(frame, f"Match: {final_conf*100:.0f}%", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "Sign Language Detection", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

if __name__ == "__main__":
    SignLanguageDetector().detect_realtime()
