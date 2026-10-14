# Edge AI Waste Classification on Arduino Nicla Vision

A six-class waste image classifier (cardboard, glass, metal, paper, plastic, trash) trained in TensorFlow/Keras, converted to a fully integer-quantized (int8) TensorFlow Lite model, and run live on an **Arduino Nicla Vision** under **OpenMV MicroPython**. The quantized model is 146 KB and has about 101k parameters. The network is a small convolutional "token-mixer" that uses only operators the OpenMV runtime supports. On the board it classifies a center crop of each camera frame and streams the predicted class, its top-class softmax score (shown as "confidence"; it is not a calibrated probability), and the loop frame rate over the serial terminal.

**Recorded results:** 77.35% Keras validation accuracy, 77.45% int8 TFLite validation accuracy, and 79.27% int8 TFLite test accuracy, all measured on the host. On the device, OpenMV IDE shows a whole-loop rate of about 14 FPS (capture, preprocessing, inference, and output) in the [recorded demo](media/demo.mp4).

The main engineering problem in this project is fitting the model to the deployment target: choosing an architecture whose operators survive full-int8 conversion, calibrating quantization against the same input range the device sends, and checking the converted model's operator set and tensor types before it goes onto a microcontroller.

## Key Technical Features

- **Deployment-constrained architecture.** The model avoids self-attention, LayerNorm, GELU/Swish, dynamic reshapes, and division/sqrt ops. The converted graph contains only `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `MEAN`, `FULLY_CONNECTED`, and `SOFTMAX`.
- **Full integer quantization.** Post-training quantization uses a 200-image representative dataset. The result has int8 input and output tensors, int8 activations and per-channel weights, and int32 biases. The graph contains no float tensors and no `QUANTIZE`/`DEQUANTIZE` ops.
- **Preprocessing inside the model.** A `Rescaling(1/255)` layer is part of the network, so the model takes raw 0–255 pixels. The converted model keeps that raw-pixel input contract. Its int8 input uses scale 1.0 and zero point −128, so each unsigned pixel value is stored in signed int8 form as the pixel value minus 128. The original rescaling is handled inside the quantized model.
- **Operator gate before deployment.** The notebook lists the converted model's operators and checks them against an OpenMV-safe allowlist.
- **Host-side int8 evaluation.** The `.tflite` model is evaluated in the TFLite interpreter on the validation and held-out test splits, with inputs quantized the same way the device quantizes them.
- **On-device inference.** The device script uses OpenMV's `csi` camera driver, `ml.Model`, and `ml.preprocessing.Normalization` with a center-crop ROI. It decodes the six-class output with an argmax and reports the result over serial.

## System Pipeline

**Development (host, `training/token_mixer_garbage.ipynb`)**

```text
images_96/<class>/*.jpg (not included)
        │  stratified 70/15/15 split per class, seed 42
        ▼
tf.data pipeline ── resize to 96×96, float32 in [0, 255], augmentation on train split only
        │
        ▼
Keras model (Rescaling 1/255 inside) ── Adam, sparse categorical cross-entropy,
        │                                EarlyStopping + ReduceLROnPlateau on val_loss
        ▼
training/c1_token_mixer.keras
        │  TFLiteConverter + representative dataset (200 clean train images)
        ▼
device/c1_token_mixer_int8.tflite ── operator check + int8 accuracy on val/test (TFLite interpreter)
```

**Runtime (Nicla Vision, `device/main.py`)**

```text
GC2145 camera, QVGA 320×240 RGB565
        │  center-crop ROI (40, 0, 240, 240)
        ▼
ml.preprocessing.Normalization ── ROI scaled to 96×96; int8 input: pixel shifted from uint8 to int8 range
        │
        ▼
ml.Model.predict ── int8 TFLite model
        │
        ▼
argmax over 6 scores ── print "Prediction: <label> | confidence: <p> | FPS: <fps>"
```

## Model Architecture

The model, `c1_token_mixer`, is built in `build_c1_token_mixer()` in the notebook. It follows the token-mixer idea from MLP-Mixer, in which each block alternates mixing across spatial positions ("tokens") with mixing across channels. The token-mixing step is implemented with convolutions so that the model stays within the OpenMV operator set:

- **Token (spatial) mixing:** a 3×3 depthwise convolution, followed by BatchNorm and ReLU, with a residual add. This mixes information between neighbouring positions instead of using a dense MLP over all patches as MLP-Mixer does. That design is conceptually closer to ConvMixer.
- **Channel mixing:** a pointwise 1×1 convolution that expands the channels by `mlp_ratio = 2`, then BatchNorm and ReLU, then a 1×1 projection back with BatchNorm, followed by a residual add.

| Stage | Operation | Output |
|---|---|---|
| Input | raw RGB, 0–255; `Rescaling(1/255)` | 96 × 96 × 3 |
| Stem | 3 × (3×3 conv, stride 2, BN, ReLU) | 48×48×12 → 24×24×24 → 12×12×48 |
| Stage 1 | 2 mixer blocks, 48 channels (hidden 96) | 12 × 12 × 48 |
| Transition | depthwise 3×3 stride 2 + pointwise 1×1 (BN, ReLU) | 6 × 6 × 80 |
| Stage 2 | 2 mixer blocks, 80 channels (hidden 160) | 6 × 6 × 80 |
| Head | global average pool → Dense 80 (ReLU) → Dense 6 (softmax) | 6 |

There are 101,402 parameters in total (98,930 trainable). The convolutions have no bias, so BatchNorm folds into them during int8 conversion.

**Training configuration (as recorded in the notebook):** input 96×96, batch size 32, Adam (lr 1e-3), sparse categorical cross-entropy, up to 80 epochs. `EarlyStopping` (patience 12, restoring the best weights) and `ReduceLROnPlateau` (factor 0.5, patience 5, minimum 1e-5) both monitor `val_loss`. Training-time augmentation consists of random horizontal flip, rotation, zoom, and contrast. It lives in the `tf.data` pipeline, not in the model, so it never reaches the exported graph.

## Model Conversion and Quantization

Conversion settings, from the notebook: `Optimize.DEFAULT`, a `representative_dataset` of 200 unaugmented training images in raw 0–255 range, `supported_ops = [TFLITE_BUILTINS_INT8]`, and `inference_input_type = inference_output_type = tf.int8`.

Static inspection of the retained `.tflite` file confirms the following:

| | Keras model | TFLite model |
|---|---|---|
| File | [`training/c1_token_mixer.keras`](training/c1_token_mixer.keras) | [`device/c1_token_mixer_int8.tflite`](device/c1_token_mixer_int8.tflite) |
| Size | 1,433,391 bytes (weights plus saved training state) | 145,928 bytes |
| Format | Keras v3 archive, Keras 2.15.0, Functional model | TFLite flatbuffer (schema v3), 1 subgraph, 29 ops |
| Input | float32, 1×96×96×3, range 0–255 | int8, 1×96×96×3, scale 1.0, zero point −128 |
| Output | float32 softmax, 1×6 | int8, 1×6, scale 1/256, zero point −128 |
| Internal tensors | float32 | 49 int8 (weights, activations), 20 int32 (biases, reduction axis); 34 tensors use per-channel scales |
| Role | training, evaluation, conversion source | deployed to the Nicla Vision |

The output quantization maps int8 values from −128 to 127 onto softmax scores from 0 to about 0.996. The `.tflite` file has no embedded label metadata, so the device script provides the class list itself.

## Arduino Nicla Vision Deployment

The device code in `device/main.py` is an **OpenMV MicroPython** script, not an Arduino sketch. It imports `csi`, `ml`, `ml.preprocessing.Normalization`, `time`, and `gc`.

**Files on the board:** `MODEL_PATH = "c1_token_mixer_int8.tflite"` is a bare filename. Both files in [`device/`](device/) therefore need to be copied together into the root of the board's OpenMV filesystem (the USB drive the firmware exposes). Then open `main.py` in OpenMV IDE and run it.

**Runtime flow:**

1. `ml.Model(MODEL_PATH)` loads the model. The script prints the model's size, its RAM requirement, the input/output shapes and dtypes, and the quantization parameters. Labels come from `model.labels` if the firmware provides them, and otherwise from `FALLBACK_LABELS`. That list is in the same alphabetical order the notebook uses for class indices.
2. The camera is reset and set to `RGB565` at `QVGA` (320×240). A 2-second snapshot lets auto-exposure settle.
3. Each loop iteration takes a snapshot and runs `Normalization(roi=(40, 0, 240, 240))`, a centered 240×240 square, which the script expects the firmware to scale to the 96×96 model input. If `ml.preprocessing` can't be imported, the script falls back to passing the full 320×240 frame.
4. `model.predict([...])` runs inference. A plain-Python argmax over the six output scores picks the class. The winning score is reported as the confidence; there is no threshold.
5. One line per frame goes to the serial terminal, for example `Prediction: metal | confidence: 0.727 | FPS: 13.97`. The FPS figure comes from `time.clock()`, so it measures the whole loop (capture, preprocessing, inference, printing). `gc.collect()` runs every iteration.

The script doesn't use LEDs, UART, or any other output besides the serial terminal.

**Preprocessing compared with training.**

- **Value range.** The model expects int8 input equal to the pixel value minus 128, because its quantization uses scale 1.0 and zero point −128 on raw 0–255 pixels. The OpenMV 4.8.1 documentation for `ml.preprocessing.Normalization` says that for `int8` input tensors, "the image is shifted to be within the `int8` range from the `uint8` range and then directly passed ignoring scale and mean/stdev." The `scale`, `mean`, and `stdev` defaults therefore don't apply to this model, and the documented shift matches the quantization. The exact pixel arithmetic was not traced in the firmware source.
- **Framing.** Training resized each whole image to 96×96. The device uses a 240×240 center crop of a 320×240 RGB565 camera frame. Live sensor images also differ from the dataset photos in lighting, background, and colour depth.
- **Effect.** How much these differences affect real-world accuracy was not systematically measured.

## Recorded Results

All numbers below come from saved notebook outputs and the demo recording. None were re-run while preparing this repository.

**Dataset split** (from notebook output): 13,901 images across 6 classes, split into 9,728 train, 2,084 validation, and 2,089 test.

| Metric | Value | Source |
|---|---|---|
| Keras validation accuracy | 0.7735 | `model.evaluate(val_ds)` after training (best weights from epoch 57 of 69) |
| int8 TFLite validation accuracy | 0.7745 | TFLite interpreter, 2,084 images |
| int8 TFLite test accuracy | 0.7927 | TFLite interpreter, 2,089 held-out images, reported once |
| Per-class F1 (Keras, validation) | 0.734 (trash) – 0.798 (glass) | `classification_report` |
| TFLite model size | 145,928 bytes | file size; also printed by the device |
| Model RAM reported by firmware | 70,224 bytes | `model.ram`, as shown in the demo recording |
| Loop rate on device | ~14 FPS | `clock.fps()` as printed in the demo recording |

The validation split was also used for early stopping and learning-rate scheduling, so the test split is the less biased of the two accuracy figures. Validation accuracy is essentially unchanged after int8 conversion on this split, but the two models were not compared prediction by prediction. The FPS figure covers one recorded session with OpenMV IDE streaming the frame buffer. It is not a controlled benchmark.

The notebook also saves training/validation accuracy and loss curves and a validation confusion matrix. In the confusion matrix, the largest off-diagonal counts are trash↔paper, paper→cardboard, cardboard→trash, and plastic↔glass.

## Demo

[Watch the demonstration](media/demo.mp4) (90 s, 2704×1756 H.264 MP4, 17.6 MB, no audio)

The screen recording shows OpenMV IDE running `main.py` on the Arduino Nicla Vision. The IDE status bar reads "Board: Arduino Nicla Vision", "Sensor: GC2145", and "Firmware Version: 4.8.1". The recording shows:

- the startup printout of the model's size, RAM, tensor shapes, dtypes, quantization parameters, and labels;
- the live camera frame buffer as objects are placed in view (a set of keys, clear packaging, and a clear glass bottle);
- streaming per-frame predictions with label, confidence, and a loop rate of about 14 FPS. The keys are mostly classified as `metal`. Predictions on the transparent objects change between `plastic`, `paper`, `glass`, and `trash`.

The physical board itself does not appear on camera; the video shows the IDE and the camera feed. The video demonstrates working on-device inference, not classification accuracy. Depending on the browser, GitHub may offer the MP4 as a download rather than playing it inline.

## Repository Structure

```text
nicla-vision-waste-classifier/
├── README.md
├── .gitignore
├── device/                          # copy both files to the board's filesystem root
│   ├── main.py                      # OpenMV MicroPython inference loop
│   └── c1_token_mixer_int8.tflite   # deployed int8 model
├── training/
│   ├── token_mixer_garbage.ipynb    # data pipeline, model, training, conversion, evaluation
│   └── c1_token_mixer.keras         # trained float Keras model (conversion source)
└── media/
    └── demo.mp4                     # on-device demo screen recording
```

`main.py` and the `.tflite` model sit together because the script loads the model by a relative filename. The notebook and the `.keras` model sit together because the conversion cell loads `c1_token_mixer.keras` from the notebook's working directory.

## Reproduction and Limitations

- **Inspecting the models:** the `.keras` file can be loaded with Keras 2.15 (the version recorded in its metadata). The `.tflite` file can be opened with any TFLite interpreter or a model viewer such as Netron.
- **Running on the device:** you need an Arduino Nicla Vision with OpenMV firmware and OpenMV IDE. The demo used firmware 4.8.1. Other firmware versions were not checked, and the script relies on the `csi` module and `ml.preprocessing`.
- **Training data is not included.** The notebook records 13,901 images in six classes, read from a local `images_96/` directory and split 70/15/15. The Kaggle garbage classification dataset is referenced as a source, but the notebook does not record how `images_96/` was assembled. It may not account for all 13,901 images, and no dataset license is recorded here. Without that directory, training and calibration can't be reproduced exactly. The retained `.keras` and `.tflite` files are the authoritative artifacts.
- **Environment:** the notebook ran on Python 3.11 on an Apple-silicon Mac (according to its logged warnings), with TensorFlow/Keras, NumPy, Matplotlib, and scikit-learn. There is no pinned requirements file.
- **Notebook state:** the saved outputs come from one top-to-bottom run. The second augmentation cell ("Stronger augmentation…") runs *after* the datasets were built, so the recorded training run most likely used the augmentation defined in the data-preparation cell. `train_one_seed()` is an unused helper from an earlier model iteration.
- **Scope:** this is an embedded inference prototype. It is not a validated waste-sorting system. Accuracy on live camera input was not measured, and the demo shows the model struggling with transparent objects.

## References

- Arduino Nicla Vision: <https://docs.arduino.cc/hardware/nicla-vision/>
- OpenMV (MicroPython machine-vision firmware and IDE): <https://openmv.io/>
- OpenMV 4.8.1 `ml.preprocessing` documentation: <https://docs.openmv.io/v4.8.1/library/omv.ml.preprocessing.html>
- TensorFlow Lite post-training integer quantization: <https://www.tensorflow.org/lite/performance/post_training_integer_quant>
- I. Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision," NeurIPS 2021. arXiv:2105.01601
- A. Trockman and J. Z. Kolter, "Patches Are All You Need?" (ConvMixer), arXiv:2201.09792
- Garbage classification dataset referenced in the notebook (provenance of the full training set not recorded): <https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification>
