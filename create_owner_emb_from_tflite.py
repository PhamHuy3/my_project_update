# create_owner_embedding_from_tflite.py
import tensorflow as tf
import numpy as np
import cv2, os

MODEL_PATH = "facenet_mcu_int8.tflite"   # hoặc facenet_mcu_fp16.tflite
DATA_DIR = "data_aligned/chu_nha"
SAVE_PATH = "owner_embedding.npy"

IMG_SIZE = (112, 112)

# ---------------- Load TFLite model ----------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale, in_zp = input_details[0]["quantization"]
out_scale, out_zp = output_details[0]["quantization"]

is_int8 = input_details[0]["dtype"] == np.int8

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE).astype("float32") / 255.0
    return img

def run_tflite(img):
    img = np.expand_dims(img, axis=0)

    if is_int8:
        img_q = img / in_scale + in_zp
        img_q = np.clip(np.round(img_q), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], img_q)
    else:
        interpreter.set_tensor(input_details[0]["index"], img.astype("float32"))

    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]["index"])[0]

    if is_int8:
        emb = (emb.astype(np.float32) - out_zp) * out_scale

    emb = emb / np.linalg.norm(emb)
    return emb

# ---------------- Create owner embedding ----------------
emb_list = []

for fname in os.listdir(DATA_DIR):
    if fname.lower().endswith((".jpg", ".png")):
        img = cv2.imread(os.path.join(DATA_DIR, fname))
        emb = run_tflite(preprocess(img))
        emb_list.append(emb)

owner_emb = np.mean(emb_list, axis=0)
owner_emb = owner_emb / np.linalg.norm(owner_emb)

np.save(SAVE_PATH, owner_emb)
print(f"[DONE] Saved embedding → {SAVE_PATH}")
