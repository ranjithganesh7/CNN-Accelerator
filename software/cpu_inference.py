import numpy as np
import cv2
import re
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
HEADER_PATH = Path("/home/xilinx/pynq/overlays/final/weights.h")
IMAGE_PATH  = Path("/home/xilinx/pynq/overlays/final/test_image_9.jpg")

CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ==========================================
# 2. MODEL DEFINITION (CPU)
# ==========================================
class SimpleCNN:
    def __init__(self, weights_path):
        self.weights = self._load_weights(weights_path)
        
        # Reshape weights to match Architecture
        self.w1 = self.weights["layer1_w"].reshape(16, 3, 3, 3)
        self.b1 = self.weights["layer1_b"]
        self.w2 = self.weights["layer2_w"].reshape(32, 16, 3, 3)
        self.b2 = self.weights["layer2_b"]
        self.w3 = self.weights["layer3_w"].reshape(64, 32, 3, 3)
        self.b3 = self.weights["layer3_b"]
        self.fc_w = self.weights["fc_w"].reshape(10, 1024)
        self.fc_b = self.weights["fc_b"]

    def _load_weights(self, filepath):
        if not filepath.exists():
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        with open(filepath, "r") as f:
            content = f.read()
        pattern = re.compile(r"static const float (\w+)\[\] = \{(.*?)\};", re.DOTALL)
        weights = {}
        for name, array_str in pattern.findall(content):
            clean = array_str.replace("\n", "").replace(" ", "")
            values = [float(x) for x in clean.split(",") if x] 
            weights[name] = np.array(values, dtype=np.float32)
        return weights

    def conv2d(self, x, weights, bias):
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

    def relu(self, x): return np.maximum(0, x)

    def max_pool(self, x, stride=2):
        h, w, ch = x.shape
        out = np.zeros((h // stride, w // stride, ch), dtype=np.float32)
        for c in range(ch):
            for y in range(0, h, stride):
                for x_ in range(0, w, stride):
                    out[y//stride, x_//stride, c] = np.max(x[y:y+stride, x_:x_+stride, c])
        return out

    def softmax(self, logits):
        e_x = np.exp(logits - np.max(logits))
        return e_x / np.sum(e_x)

    def forward(self, img_tensor):
        x = self.max_pool(self.relu(self.conv2d(img_tensor, self.w1, self.b1)))
        x = self.max_pool(self.relu(self.conv2d(x, self.w2, self.b2)))
        x = self.max_pool(self.relu(self.conv2d(x, self.w3, self.b3)))
        x_flat = x.transpose(2, 0, 1).flatten()
        return self.softmax(np.dot(self.fc_w, x_flat) + self.fc_b)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    print("🚀 Starting CPU Inference...\n")
    
    # --- Setup ---
    model = SimpleCNN(HEADER_PATH)
    if not IMAGE_PATH.exists(): raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # --- Preprocessing ---
    img_bgr = cv2.imread(str(IMAGE_PATH))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (32, 32))
    
    # Normalize
    img_tensor = (img_resized / 255.0 - 0.5) / 0.5
    img_tensor = img_tensor.astype(np.float32)

    # --- Display Input Images ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Test Image")
    axes[0].axis('off')
    
    axes[1].imshow(img_resized)
    axes[1].set_title("CPU Input (32x32)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # --- Run Inference ---
    start = time.perf_counter()
    probs = model.forward(img_tensor)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    # --- Results ---
    top5 = np.argsort(probs)[::-1][:5]
    top_label = CLASSES[top5[0]].upper()
    top_score = probs[top5[0]] * 100

    print(f"⏱️ Inference Time: {latency_ms:.2f} ms")
    print("\n-------------------------------------------")
    print("📊 TOP 5 PREDICTIONS (CPU SOFTWARE):")
    print("-------------------------------------------")
    for i, idx in enumerate(top5):
        print(f"{i+1}. {CLASSES[idx].upper():<10} : {probs[idx]*100:>6.2f}%")
    print("-------------------------------------------")

    # --- Display Final Result ---
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_rgb)
    h, w, _ = img_rgb.shape

    # Draw bounding box
    rect = patches.Rectangle((1, 1), w-2, h-2, linewidth=5, 
                             edgecolor='#00FF00', facecolor='none')
    ax.add_patch(rect)

    # Draw label tag
    tag = f" {top_label}: {top_score:.1f}% "
    ax.text(w * 0.02, h * 0.08, tag, color='white', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='#00FF00', edgecolor='none', pad=0.3))

    plt.title("CPU Inference Result", fontsize=15)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
