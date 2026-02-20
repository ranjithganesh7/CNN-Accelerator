from pynq import Overlay, allocate, MMIO
import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
BITSTREAM = "/home/xilinx/pynq/overlays/detection/design_1_wrapper.bit"
IMAGE_PATH = "/home/xilinx/pynq/overlays/detection/test_image.jpg"
W_IN, H_IN = 320, 320
W_OUT, H_OUT = 160, 160

if not os.path.exists(BITSTREAM): raise FileNotFoundError("Bitstream not found")
overlay = Overlay(BITSTREAM)
dma = overlay.axi_dma_0

if 'dpu_core_0' in overlay.ip_dict:
    dpu_addr = overlay.ip_dict['dpu_core_0']['phys_addr']
else:
    for ip in overlay.ip_dict:
        if 'dpu' in ip:
            dpu_addr = overlay.ip_dict[ip]['phys_addr']
            break
dpu = MMIO(dpu_addr, 65536)

# Load Weights
weights_list = [0.0, -1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0]
weights_buf = allocate(shape=(1024,), dtype=np.float32)
np.copyto(weights_buf[:10], np.array(weights_list, dtype=np.float32))
weights_buf.flush() 
dpu.write(0x10, weights_buf.device_address)

# Load Image
img_original = cv2.imread(IMAGE_PATH)
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img_gray, (W_IN, H_IN))
img_float = img_resized.astype(np.float32)

# ==========================================
# 2. HELPER: DETECTION LOGIC
# ==========================================
def detect_and_draw(feature_map, display_img, color, label_prefix):
    # Normalize & Threshold
    fm = np.abs(feature_map)
    fm = cv2.normalize(fm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(fm, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological Clean up
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find Contours
    _, contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_img = display_img.copy()
    label = "No Object"
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        scale = 2 
        x *= scale; y *= scale; w *= scale; h *= scale
        
        cv2.rectangle(final_img, (x, y), (x + w, y + h), color, 2)
        label = label_prefix
        
    return final_img, label

# ==========================================
# 3. RUN FPGA (HARDWARE)
# ==========================================
print("🚀 Running FPGA (100 loops)...")
in_buf = allocate(shape=(H_IN * W_IN,), dtype=np.float32)
out_buf = allocate(shape=(H_OUT * W_OUT,), dtype=np.float32)
np.copyto(in_buf, img_float.flatten())
in_buf.flush()

start_fpga = time.time()
for i in range(100):
    dpu.write(0x1C, W_IN); dpu.write(0x24, H_IN); dpu.write(0x00, 0x81)
    dma.sendchannel.transfer(in_buf)
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
end_fpga = time.time()

avg_fpga_ms = ((end_fpga - start_fpga) / 100) * 1000
fps_fpga = 1000.0 / avg_fpga_ms

# Get Result for Display
out_buf.invalidate()
fpga_result = np.array(out_buf).reshape(H_OUT, W_OUT)
img_fpga_disp = cv2.resize(img_original, (W_IN, H_IN))
img_fpga_final, label_fpga = detect_and_draw(fpga_result, img_fpga_disp, (0, 255, 0), f"FPGA: {fps_fpga:.0f} FPS")

# ==========================================
# 4. RUN CPU (SOFTWARE)
# ==========================================
print("🐌 Running CPU (20 loops)...")
def run_cpu_inference(image, kernel):
    k_mat = np.array(kernel[1:]).reshape(3,3)
    bias = kernel[0]
    conv_out = cv2.filter2D(image, -1, k_mat) + bias
    relu_out = np.maximum(conv_out, 0)
    h, w = relu_out.shape
    pool_out = relu_out.reshape(h//2, 2, w//2, 2).max(axis=(1, 3))
    return pool_out

start_cpu = time.time()
for i in range(20):
    cpu_result = run_cpu_inference(img_float, weights_list)
end_cpu = time.time()

avg_cpu_ms = ((end_cpu - start_cpu) / 20) * 1000
fps_cpu = 1000.0 / avg_cpu_ms

# Get Result for Display
img_cpu_disp = cv2.resize(img_original, (W_IN, H_IN))
# Use Red Box for CPU (0, 0, 255)
img_cpu_final, label_cpu = detect_and_draw(cpu_result, img_cpu_disp, (255, 0, 0), f"CPU: {fps_cpu:.0f} FPS")

# ==========================================
# 5. FINAL COMPARISON PLOT
# ==========================================
plt.figure(figsize=(18, 6))

# 1. Bar Chart
plt.subplot(1, 3, 1)
labels = ['FPGA', 'CPU']
times = [avg_fpga_ms, avg_cpu_ms]
bars = plt.bar(labels, times, color=['#4CAF50', '#F44336'])
plt.ylabel('Latency (ms)')
plt.title(f'Speed Comparison\nFPGA is {avg_cpu_ms/avg_fpga_ms:.1f}x Faster')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f} ms', ha='center', va='bottom')

# 2. FPGA Result
plt.subplot(1, 3, 2)
plt.title(label_fpga + " (Green Box)")
plt.imshow(cv2.cvtColor(img_fpga_final, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 3. CPU Result
plt.subplot(1, 3, 3)
plt.title(label_cpu + " (Red Box)")
plt.imshow(cv2.cvtColor(img_cpu_final, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# Clean up
in_buf.close(); out_buf.close(); weights_buf.close()
