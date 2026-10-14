# Garbage Classification on Nicla Vision
# Model: 96x96 int8 TFLite image classifier

import csi
import time
import ml
import gc

try:
    from ml.preprocessing import Normalization
except Exception:
    Normalization = None


MODEL_PATH = "c1_token_mixer_int8.tflite"

# These labels are used only if the .txt label file is not loaded automatically.
FALLBACK_LABELS = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# QVGA is 320x240. This ROI center-crops a 240x240 square.
# The ML preprocessing step will resize it to the model input size 96x96.
CENTER_CROP_ROI = (40, 0, 240, 240)


def get_score(output, index):
    # Handle both flat output shape (6,) and batched output shape (1, 6).
    try:
        return float(output[0][index])
    except Exception:
        return float(output[index])


def argmax_output(output, num_classes):
    # Find the class index with the highest score.
    best_index = 0
    best_score = get_score(output, 0)

    for i in range(1, num_classes):
        score = get_score(output, i)
        if score > best_score:
            best_score = score
            best_index = i

    return best_index, best_score


def print_top_prediction(labels, pred_index, confidence, fps):
    # Print one compact line for the OpenMV serial terminal.
    label = labels[pred_index]
    print("Prediction: %s | confidence: %.3f | FPS: %.2f" % (label, confidence, fps))


# Load model
model = ml.Model(MODEL_PATH)

# Load labels automatically if possible.
labels = model.labels if model.labels else FALLBACK_LABELS
num_classes = len(labels)

print("Model loaded")
print("Model size bytes:", model.len)
print("Model RAM bytes:", model.ram)
print("Input shape:", model.input_shape)
print("Input dtype:", model.input_dtype)
print("Input scale:", model.input_scale)
print("Input zero point:", model.input_zero_point)
print("Output shape:", model.output_shape)
print("Output dtype:", model.output_dtype)
print("Output scale:", model.output_scale)
print("Output zero point:", model.output_zero_point)
print("Labels:", labels)

# Initialize camera
cam = csi.CSI()
cam.reset()
cam.pixformat(csi.RGB565)
cam.framesize(csi.QVGA)

# Let camera settings settle
cam.snapshot(time=2000)

clock = time.clock()

# Create preprocessing object if supported by the firmware.
if Normalization:
    preprocess = Normalization(roi=CENTER_CROP_ROI)
else:
    preprocess = None
    print("Normalization import failed. Falling back to full-frame image input.")

while True:
    clock.tick()

    img = cam.snapshot()

    # Use center crop if available. Otherwise, pass the full image directly.
    if preprocess:
        outputs = model.predict([preprocess(img)])
    else:
        outputs = model.predict([img])

    output = outputs[0]

    pred_index, confidence = argmax_output(output, num_classes)

    print_top_prediction(labels, pred_index, confidence, clock.fps())

    gc.collect()
