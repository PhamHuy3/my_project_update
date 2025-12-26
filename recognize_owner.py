import cv2
import numpy as np
import tensorflow as tf
import time
import os
import sys


DETECTOR_PATH = "face_detection_front.tflite"
MODEL_PATH    = "facenet_mcu_int8.tflite"
OWNER_EMB_PATH = "owner_embedding.npy"

IMG_SIZE = (112, 112)
THRESHOLD = 0.65
CAM_INDEX = 0
DET_CONF_THR = 0.5   
MARGIN = 0.25        
SHOW_FPS = True


# load detector tflite
if not os.path.exists(DETECTOR_PATH):
    raise FileNotFoundError(f"Detector not found: {DETECTOR_PATH}")
detector = tf.lite.Interpreter(model_path=DETECTOR_PATH)
detector.allocate_tensors()
det_in = detector.get_input_details()
det_out = detector.get_output_details()

det_input_q = det_in[0].get("quantization", (0.0,0))
det_scale, det_zp = det_input_q if isinstance(det_input_q, (list,tuple)) and len(det_input_q)>=2 else (0.0,0)
det_dtype = det_in[0]["dtype"]
det_input_h = det_in[0]['shape'][1]
det_input_w = det_in[0]['shape'][2]

# load facenet tflite
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"FaceNet model not found: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

in_q = in_details[0].get("quantization", (0.0,0))
in_scale, in_zp = in_q if isinstance(in_q, (list,tuple)) and len(in_q)>=2 else (0.0,0)
in_dtype = in_details[0]["dtype"]
out_q = out_details[0].get("quantization", (0.0,0))
out_scale, out_zp = out_q if isinstance(out_q, (list,tuple)) and len(out_q)>=2 else (0.0,0)
model_h = in_details[0]['shape'][1]; model_w = in_details[0]['shape'][2]
is_input_int8 = np.issubdtype(in_dtype, np.integer)
is_output_int = np.issubdtype(out_details[0]['dtype'], np.integer)

# load owner embedding
if not os.path.exists(OWNER_EMB_PATH):
    raise FileNotFoundError(f"Owner embedding not found: {OWNER_EMB_PATH}. Run create_owner_embedding script first.")
owner_emb = np.load(OWNER_EMB_PATH).astype(np.float32)
nrm = np.linalg.norm(owner_emb) + 1e-12
owner_emb = owner_emb / nrm


def preprocess_blazeface(frame):
    img = cv2.resize(frame, (det_input_w, det_input_h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if det_scale != 0:
        # quantized model
        arr = img_rgb.astype(np.float32) / det_scale + det_zp
        arr = np.clip(np.round(arr), np.iinfo(det_dtype).min, np.iinfo(det_dtype).max).astype(det_dtype)
        return np.expand_dims(arr, 0)
    else:
        # float model: normalize to [-1,1] (common for BlazeFace)
        arr = img_rgb.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return np.expand_dims(arr, 0).astype(np.float32)

def detect_face(frame):
    inp = preprocess_blazeface(frame)
    detector.set_tensor(det_in[0]["index"], inp)
    detector.invoke()

    raw_boxes = detector.get_tensor(det_out[0]["index"])[0]  
    scores    = detector.get_tensor(det_out[1]["index"])[0]  

    idx = np.argmax(scores)
    score = float(scores[idx])

    if score < DET_CONF_THR:
        return None

    cx, cy, w, h = raw_boxes[idx][:4]

    H, W = frame.shape[:2]
    x1 = int((cx - w/2) * W)
    y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W)
    y2 = int((cy + h/2) * H)

    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2))
    y2 = max(0, min(H-1, y2))

    return x1, y1, x2, y2, score



def expand_to_square(x1, y1, x2, y2, img_w, img_h, margin=MARGIN):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    side = max(w, h) * (1.0 + margin)
    nx1 = int(max(0, cx - side/2))
    ny1 = int(max(0, cy - side/2))
    nx2 = int(min(img_w, cx + side/2))
    ny2 = int(min(img_h, cy + side/2))
    return nx1, ny1, nx2, ny2

def preprocess_facenet(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (model_w, model_h)).astype(np.float32) / 255.0
    inp = np.expand_dims(face_resized, 0)  # (1,H,W,3)
    if is_input_int8 and in_scale != 0:
        q = inp / in_scale + in_zp
        q = np.round(q)
        if np.issubdtype(in_dtype, np.unsignedinteger):
            q = np.clip(q, 0, 255).astype(in_dtype)
        else:
            q = np.clip(q, -128, 127).astype(in_dtype)
        return q
    return inp.astype(np.float32)

def get_embedding(face_bgr):
    inp = preprocess_facenet(face_bgr)
    interpreter.set_tensor(in_details[0]['index'], inp)
    interpreter.invoke()
    emb = interpreter.get_tensor(out_details[0]['index'])[0]
    if is_output_int and out_scale != 0:
        emb = (emb.astype(np.float32) - out_zp) * out_scale
    emb = emb.astype(np.float32)
    n = np.linalg.norm(emb) + 1e-12
    emb = emb / n
    return emb

def cosine(a,b):
    return float(np.dot(a,b) / ((np.linalg.norm(a)+1e-12)*(np.linalg.norm(b)+1e-12)))

# CAMERA LOOP
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("[INFO] Start STM32N6-compatible test. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = detect_face(frame)
    if res is not None:
        x1, y1, x2, y2, conf = res
        img_h, img_w = frame.shape[:2]

        
        sx1, sy1, sx2, sy2 = expand_to_square(
            x1, y1, x2, y2, img_w, img_h, margin=MARGIN
        )
        face = frame[sy1:sy2, sx1:sx2].copy()

        if face.size != 0:
            try:
                emb = get_embedding(face)
                sim = cosine(emb, owner_emb)
            except Exception as e:
                print("Embedding error:", e)
                emb = None
                sim = 0.0

            
            if emb is not None and sim >= THRESHOLD:
                label = "CHU NHA"
                color = (0, 255, 0)
            else:
                label = "NGUOI LA"
                color = (0, 0, 255)

        
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)

            
            text_y = sy1 - 10
            if text_y < 15:
                text_y = sy1 + 20
            cv2.putText(frame, f"{label} {sim:.2f}",
                        (sx1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imshow("STM32N6 Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
