import numpy as np
import cv2
import re
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

HEADER_PATH = Path("./weights.h")
IMAGE_PATH  = Path("./test_image_1.jpg") 

CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ------------------------------------------------------------
# Weight Loader
# ------------------------------------------------------------

def load_weights(filepath: Path) -> dict:
    if not filepath.exists():
        raise FileNotFoundError(f"Weights file not found: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    pattern = re.compile(
        r"static const float (\w+)\[\] = \{(.*?)\};",
        re.DOTALL
    )

    weights = {}
    for name, array_str in pattern.findall(content):
        clean = array_str.replace("\n", "").replace(" ", "")
        values = [float(x) for x in clean.split(",") if x]
        weights[name] = np.array(values, dtype=np.float32)

    return weights


# ------------------------------------------------------------
# Core Operations
# ------------------------------------------------------------

def conv2d(x, weights, bias):
    h, w, ch_in = x.shape
    ch_out = weights.shape[0]

    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="constant")
    out = np.zeros((h, w, ch_out), dtype=np.float32)

    for co in range(ch_out):
        for ci in range(ch_in):
            kernel = weights[co, ci]
            for i in range(3):
                for j in range(3):
                    out[:, :, co] += padded[i:i+h, j:j+w, ci] * kernel[i, j]
        out[:, :, co] += bias[co]

    return out


def relu(x):
    return np.maximum(0, x)


def max_pool(x, stride=2):
    h, w, ch = x.shape
    out = np.zeros((h // stride, w // stride, ch), dtype=np.float32)

    for c in range(ch):
        for y in range(0, h, stride):
            for x_ in range(0, w, stride):
                window = x[y:y+stride, x_:x_+stride, c]
                out[y//stride, x_//stride, c] = np.max(window)

    return out


def softmax(logits):
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    weights = load_weights(HEADER_PATH)

    w1 = weights["layer1_w"].reshape(16, 3, 3, 3)
    b1 = weights["layer1_b"]

    w2 = weights["layer2_w"].reshape(32, 16, 3, 3)
    b2 = weights["layer2_b"]

    w3 = weights["layer3_w"].reshape(64, 32, 3, 3)
    b3 = weights["layer3_b"]

    fc_w = weights["fc_w"].reshape(10, 1024)
    fc_b = weights["fc_b"]

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    img = cv2.imread(str(IMAGE_PATH))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (32, 32))

    img_tensor = (img_resized / 255.0 - 0.5) / 0.5
    img_tensor = img_tensor.astype(np.float32)

    start = time.perf_counter()

    x = conv2d(img_tensor, w1, b1)
    x = relu(x)
    x = max_pool(x)

    x = conv2d(x, w2, b2)
    x = relu(x)
    x = max_pool(x)

    x = conv2d(x, w3, b3)
    x = relu(x)
    x = max_pool(x)

    x = x.transpose(2, 0, 1).flatten()
    logits = np.dot(fc_w, x) + fc_b
    probs = softmax(logits)

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    top5 = np.argsort(probs)[::-1][:5]

    print("\n========================================")
    print(f"Inference Time : {latency_ms:.2f} ms")
    print("----------------------------------------")
    print(f"{'Rank':<6}{'Class':<12}{'Confidence'}")
    print("----------------------------------------")

    for rank, idx in enumerate(top5):
        print(f"{rank+1:<6}{CLASSES[idx]:<12}{probs[idx]*100:>8.2f}%")

    print("========================================\n")

    best_idx = top5[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_resized)

    rect = patches.Rectangle(
        (0.5, 0.5),
        30.5,
        30.5,
        linewidth=3,
        edgecolor="lime",
        facecolor="none"
    )

    ax.add_patch(rect)
    ax.set_title(
        f"Prediction: {CLASSES[best_idx].upper()} "
        f"({probs[best_idx]*100:.2f}%)"
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
