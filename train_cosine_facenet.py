# train_and_convert_facenet_mcu.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os, random
from pathlib import Path

# ----------------- CẤU HÌNH -----------------
IMG_SIZE = (112, 112)
BATCH_SIZE = 16
EPOCHS = 60
DATA_DIR = "data_aligned"   # expects subfolders "chu_nha" and "nguoi_la"
EMBED_DIM = 512
MARGIN = 0.5
REP_SAMPLES = 1000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------- Mô hình tối giản, tương thích MCU -----------------
def build_facenet_mcu(img_size=IMG_SIZE, emb_dim=EMBED_DIM):
    inp = layers.Input((*img_size, 3), name="input_image")

    # Conv block 1
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=True)(inp)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)  # 56x56

    # Conv block 2
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=True)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)  # 28x28

    # Conv block 3
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=True)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)  # 14x14

    # Conv block 4 (further spatial reduction)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', use_bias=True)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)  # 7x7

    x = layers.GlobalAveragePooling2D()(x)  # -> 256

    x = layers.Dense(emb_dim, use_bias=True)(x)
    # L2 normalize to produce stable embedding
    x = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name="emb_l2norm")(x)

    model = models.Model(inputs=inp, outputs=x, name="facenet_mcu")
    return model

base_model = build_facenet_mcu()

# ----------------- Cosine embedding loss -----------------
def cosine_embedding_loss(y_true, y_pred, margin=MARGIN):
    emb_dim = tf.shape(y_pred)[1] // 2
    emb1 = y_pred[:, :emb_dim]
    emb2 = y_pred[:, emb_dim:]

    emb1 = tf.math.l2_normalize(emb1, axis=1)
    emb2 = tf.math.l2_normalize(emb2, axis=1)
    cosine_sim = tf.reduce_sum(emb1 * emb2, axis=1)
    y_true = tf.squeeze(y_true)

    pos_loss = 1.0 - cosine_sim
    neg_loss = tf.maximum(0.0, cosine_sim - margin)
    loss = tf.where(y_true > 0, pos_loss, neg_loss)
    return tf.reduce_mean(loss)

# ----------------- Pair batch generator -----------------
def load_pair_batch(data_dir, batch_size=BATCH_SIZE):
    data_dir = Path(data_dir)
    chu_nha = list((data_dir / "chu_nha").glob("*.*"))
    nguoi_la = list((data_dir / "nguoi_la").glob("*.*"))

    if len(chu_nha) == 0 or len(nguoi_la) == 0:
        raise RuntimeError("Không tìm thấy ảnh trong data_aligned/chu_nha hoặc /nguoi_la")

    def preprocess(path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(img)
        arr = arr / 255.0
        return arr.astype(np.float32)

    while True:
        imgs1, imgs2, labels = [], [], []
        for _ in range(batch_size):
            if random.random() < 0.5 and len(chu_nha) >= 2:
                a, b = random.sample(chu_nha, 2)
                label = 1.0
            else:
                a = random.choice(chu_nha)
                b = random.choice(nguoi_la)
                label = -1.0
            imgs1.append(preprocess(a))
            imgs2.append(preprocess(b))
            labels.append(label)
        yield [np.array(imgs1), np.array(imgs2)], np.array(labels).reshape(-1,1)

# ----------------- Siamese model -----------------
input1 = layers.Input(shape=(*IMG_SIZE, 3), name="in1")
input2 = layers.Input(shape=(*IMG_SIZE, 3), name="in2")
emb1 = base_model(input1)
emb2 = base_model(input2)
merged = layers.Concatenate(axis=1)([emb1, emb2])
siamese = models.Model(inputs=[input1, input2], outputs=merged)

siamese.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=cosine_embedding_loss)

# ----------------- Train -----------------
train_gen = load_pair_batch(DATA_DIR, batch_size=BATCH_SIZE)
steps_per_epoch = 30

print("[INFO] Training model...")
siamese.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)

# ----------------- Save base model -----------------
base_model.save("facenet_mcu_base.h5")
print("[INFO] Saved Keras base model 'facenet_mcu_base.h5'")

# ----------------- Representative dataset for quantization -----------------
def representative_data_gen(data_dir=DATA_DIR, num_samples=REP_SAMPLES):
    data_dir = Path(data_dir)
    all_images = list((data_dir / "chu_nha").glob("*.*")) + list((data_dir / "nguoi_la").glob("*.*"))
    if len(all_images) == 0:
        raise RuntimeError("No images found for representative dataset.")
    random.shuffle(all_images)
    selected = all_images[:num_samples]
    for p in selected:
        img = tf.keras.preprocessing.image.load_img(p, target_size=IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr.astype(np.float32), axis=0)
        yield [arr]

# ----------------- Convert to TFLite FULL INT8 -----------------
def convert_to_tflite_int8(keras_model, out_file="facenet_mcu_int8.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(DATA_DIR)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_file, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Saved INT8 tflite: {out_file}")
    return out_file

tflite_int8 = convert_to_tflite_int8(base_model, out_file="facenet_mcu_int8.tflite")

# ----------------- Optional FP16 fallback -----------------
def convert_to_tflite_float16(keras_model, out_file="facenet_mcu_fp16.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(out_file, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Saved FP16 tflite: {out_file}")
    return out_file

tflite_fp16 = convert_to_tflite_float16(base_model, out_file="facenet_mcu_fp16.tflite")

# ----------------- Export C header for embedding into STM32 project -----------------
def tflite_to_c_header(tflite_file, header_file="facenet_mcu_int8_model.h", array_name="facenet_mcu_model"):
    with open(tflite_file, "rb") as f:
        data = f.read()
    # write as unsigned char array
    with open(header_file, "w") as h:
        h.write("// Auto-generated from " + os.path.basename(tflite_file) + "\n")
        h.write("#ifndef " + array_name.upper() + "_H\n")
        h.write("#define " + array_name.upper() + "_H\n\n")
        h.write(f"const unsigned char {array_name}[] = {{\n")
        for i, b in enumerate(data):
            if i % 12 == 0:
                h.write("  ")
            h.write(str(b) + ",")
            if (i + 1) % 12 == 0:
                h.write("\n")
        h.write("\n};\n")
        h.write(f"const unsigned int {array_name}_len = {len(data)};\n\n")
        h.write("#endif\n")
    print(f"[INFO] Wrote C header: {header_file}")

tflite_to_c_header("facenet_mcu_int8.tflite", header_file="facenet_mcu_int8_model.h", array_name="facenet_mcu_model")

# ----------------- Quick validate -----------------
def quick_validate(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    sample = next(representative_data_gen(DATA_DIR, num_samples=1))[0]
    if input_details[0]['dtype'] == np.int8:
        scale, zp = input_details[0]['quantization']
        sample_q = (sample / scale + zp).round().astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], sample_q)
    else:
        interpreter.set_tensor(input_details[0]['index'], sample.astype(np.float32))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    print("[INFO] Quick validate output shape:", out.shape)

try:
    quick_validate(tflite_int8)
except Exception as e:
    print("[WARN] Quick validate failed:", e)

print("[DONE] Files: facenet_mcu_base.h5, facenet_mcu_int8.tflite, facenet_mcu_fp16.tflite, facenet_mcu_int8_model.h")
