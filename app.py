import cv2
import pickle
import numpy as np
import mediapipe as mp  # pyright: ignore[reportMissingImports]
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)
import av  # pyright: ignore[reportMissingImports]
import json
import time
from threading import Lock
from typing import Optional, Tuple

# ----------------------------
# Model loading & preprocessing
# ----------------------------

@st.cache_resource
def load_model(model_path: str = "models/sign_language_model.pkl"):
    """
    Load trained Random Forest model and label map from pickle.
    """
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    labels_map = model_data["labels_map"]
    return model, labels_map


def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks.
    """
    coords = []
    for landmark in landmarks:
        coords.extend([landmark.x, landmark.y, landmark.z])

    coords = np.array(coords).reshape(-1, 3)
    wrist = coords[0]
    coords = coords - wrist

    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist

    return coords.flatten()


def predict_sign(model, labels_map, landmarks):
    """
    Run prediction on a single-hand landmark list.
    """
    feat = normalize_landmarks(landmarks).reshape(1, -1)
    pred_idx = model.predict(feat)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)[0][pred_idx]
    else:
        proba = 1.0

    label = labels_map[pred_idx]
    return label, float(proba)


# ----------------------------
# Video processor for WebRTC
# ----------------------------

class SignLanguageVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.model, self.labels_map = load_model()

        self._lock = Lock()
        self.latest_label: Optional[str] = None
        self.latest_conf: Optional[float] = None
        self.latest_ts: float = 0.0

    def get_latest(self) -> Tuple[Optional[str], Optional[float], float]:
        with self._lock:
            return self.latest_label, self.latest_conf, self.latest_ts

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        label_text = ""
        conf_text = ""

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            self.mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            try:
                label, conf = predict_sign(
                    self.model, self.labels_map, hand_landmarks.landmark
                )
                label_text = f"Sign: {label}"
                conf_text = f"Confidence: {conf * 100:.1f}%"
                with self._lock:
                    self.latest_label = str(label)
                    self.latest_conf = float(conf)
                    self.latest_ts = time.time()
            except Exception:
                label_text = "Processing..."
                conf_text = ""
                with self._lock:
                    self.latest_label = None
                    self.latest_conf = None
                    self.latest_ts = time.time()
        else:
            with self._lock:
                self.latest_label = None
                self.latest_conf = None
                self.latest_ts = time.time()

        # --- Modern UI Overlay on Video ---
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (320, 90), (20, 20, 20), -1)
        # Add slight transparency to the background box
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Professional font setup
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if label_text:
            # Green text for prediction
            cv2.putText(img, label_text, (25, 45), font, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
            if conf_text:
                # White/Gray text for confidence
                cv2.putText(img, conf_text, (25, 75), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Waiting for hand...", (25, 55), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------
# Streamlit app layout
# ----------------------------

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def speak_in_browser(text: str, lang: str = "en-US", rate: float = 1.0, volume: float = 1.0):
    """
    Trigger speech on the client browser using Web Speech API.
    Note: This only works in browsers that support speechSynthesis.
    """
    safe_text = json.dumps(text)
    safe_lang = json.dumps(lang)
    safe_rate = json.dumps(rate)
    safe_vol = json.dumps(volume)
    components.html(
        f"""
        <script>
        (function() {{
            const text = {safe_text};
            const lang = {safe_lang};
            const rate = {safe_rate};
            const volume = {safe_vol};
            if (!text) return;
            if (!('speechSynthesis' in window)) return;

            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.lang = lang;
            u.rate = rate;
            u.volume = Math.max(0, Math.min(1, volume));
            window.speechSynthesis.speak(u);
        }})();
        </script>
        """,
        height=0,
    )


def main():
    st.set_page_config(page_title="AI Sign Language Recognition", page_icon="🤖", layout="centered")

    # --- Modern Professional Theme (CSS) ---
    st.markdown(
        """
        <style>
        /* Import clean modern fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Kanit:wght@300;400;500&display=swap');

        /* App Background */
        .stApp {
            background-color: #0e1117; /* Sleek dark background */
            font-family: 'Inter', 'Kanit', sans-serif;
            color: #f8f9fa;
        }

        /* Modern Gradient Title */
        .modern-title {
            text-align: center;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }

        .modern-subtitle {
            text-align: center;
            color: #a1a1aa;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* Glassmorphism Info Panel */
        .glass-panel {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }

        .glass-panel h3 {
            color: #4facfe;
            margin-top: 0;
            font-size: 1.2rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }

        .glass-panel ul {
            margin: 0;
            padding-left: 1.5rem;
            color: #d4d4d8;
        }

        .glass-panel li {
            margin-bottom: 0.5rem;
            line-height: 1.5;
        }

        /* Video Container Styling */
        video {
            border-radius: 16px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            max-width: 100% !important;
            height: auto !important;
        }

        /* Responsive tweaks for mobile */
        @media (max-width: 768px) {
            .modern-title {
                font-size: 2rem;
            }
            .modern-subtitle {
                font-size: 0.95rem;
                margin-bottom: 1.2rem;
            }
            .glass-panel {
                padding: 1.25rem 1.5rem;
                margin-bottom: 1.2rem;
            }
        }

        /* Hide elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Layout Structure ---
    st.markdown('<div class="modern-title">Sign Language Recognition</div>', unsafe_allow_html=True)

    

    st.markdown(
        """
        <div class="glass-panel">
            <h3>⚙️ System Instructions</h3>
            <ul>
                <li>Please allow camera access when prompted by your browser.</li>
                <li>Position your hand clearly within the camera frame.</li>
                <li>The AI model (Random Forest) will translate your signs in real-time.</li>
                <li>Predictions and confidence scores will be displayed directly on the video feed.</li>
                <li><b>On mobile</b>: Open this page in your phone browser (same Wi‑Fi), allow camera, and hold the phone in portrait mode for best experience.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Speech Controls (Dropdown/Expander) ---
    # Defaults in session_state
    ss = st.session_state
    ss.setdefault("cfg_enable_speech", True)
    ss.setdefault("cfg_lang", "en-US")
    ss.setdefault("cfg_min_conf", 0.70)
    ss.setdefault("cfg_cooldown", 1.5)
    ss.setdefault("cfg_change_only", True)
    ss.setdefault("cfg_refresh_ms", 250)
    ss.setdefault("cfg_rate", 1.0)
    ss.setdefault("cfg_volume", 1.0)

    with st.expander("🎙️ Speech Controls", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            ss.cfg_enable_speech = st.toggle("Enable speech", value=ss.cfg_enable_speech)
            ss.cfg_lang = st.selectbox("Language", ["en-US", "th-TH"], index=0 if ss.cfg_lang == "en-US" else 1)
        with col2:
            ss.cfg_volume = st.slider("Volume", 0.0, 1.0, float(ss.cfg_volume), 0.05)
            ss.cfg_min_conf = st.slider("Min confidence", 0.0, 1.0, float(ss.cfg_min_conf), 0.05)
            ss.cfg_cooldown = st.slider("Cooldown (s)", 0.0, 5.0, float(ss.cfg_cooldown), 0.1)
            ss.cfg_rate = st.slider("Rate", 0.5, 1.5, float(ss.cfg_rate), 0.1)

        ss.cfg_change_only = st.toggle("Speak only when label changes", value=ss.cfg_change_only)
        ss.cfg_refresh_ms = st.slider("UI refresh (ms)", 100, 1000, int(ss.cfg_refresh_ms), 50)

    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="sign-language-recognition",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            # Use front camera on mobile if available, but keep it "ideal"
            # so that browsers can still choose any available camera.
            "video": {"facingMode": {"ideal": "user"}},
            "audio": False,
        },
        video_processor_factory=SignLanguageVideoProcessor,
    )

    if "last_spoken_label" not in st.session_state:
        st.session_state.last_spoken_label = None
    if "last_spoken_ts" not in st.session_state:
        st.session_state.last_spoken_ts = 0.0

    status = st.empty()
    speak_slot = st.empty()

    # Periodically pull latest prediction from video processor and speak it
    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        while True:
            if not webrtc_ctx.state.playing:
                break

            vp = webrtc_ctx.video_processor
            label, conf, ts = vp.get_latest() if vp else (None, None, 0.0)

            if label and conf is not None:
                status.markdown(
                    f"**Current sign**: `{label}`  \n**Confidence**: `{conf*100:.1f}%`"
                )

                now = time.time()
                enough_conf = conf >= float(st.session_state.cfg_min_conf)
                cooldown_ok = (now - float(st.session_state.last_spoken_ts)) >= float(
                    st.session_state.cfg_cooldown
                )
                changed_ok = (not st.session_state.cfg_change_only) or (
                    label != st.session_state.last_spoken_label
                )

                if st.session_state.cfg_enable_speech and enough_conf and cooldown_ok and changed_ok:
                    # Speak via browser
                    with speak_slot:
                        speak_in_browser(
                            label,
                            lang=st.session_state.cfg_lang,
                            rate=float(st.session_state.cfg_rate),
                            volume=float(st.session_state.cfg_volume),
                        )
                    st.session_state.last_spoken_label = label
                    st.session_state.last_spoken_ts = now
            else:
                status.info("Show your hand to the camera to start recognition...")

            time.sleep(float(st.session_state.cfg_refresh_ms) / 1000.0)


if __name__ == "__main__":
    main()