from pynq import Overlay, allocate, MMIO
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
# Load bitstream and DMA
ol = Overlay("/home/xilinx/pynq/overlays/final/design_1_wrapper.bit")
dma = ol.axi_dma_0

# Map the CNN IP
ip_info = ol.ip_dict['cnn_accelerator_0']
cnn_ip = MMIO(ip_info['phys_addr'], ip_info['addr_range'])

# Allocate memory buffers for input (32x32x3 = 3072) and output (10 classes)
in_buffer = allocate(shape=(3072,), dtype=np.float32)
out_buffer = allocate(shape=(10,), dtype=np.float32)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==========================================
# 2. INFERENCE FUNCTION
# ==========================================
def run_fpga_inference(img_path):
    print("🚀 Starting FPGA Inference...\n")
    
    # --- Image Preprocessing ---
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (32, 32))
    
    # Normalize image to [-1, 1] and load into DMA buffer
    img_tensor = (img_resized / 255.0 - 0.5) / 0.5
    np.copyto(in_buffer, img_tensor.astype(np.float32).flatten())
    
    # --- Display Input Images ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Test Image")
    axes[0].axis('off')
    
    axes[1].imshow(img_resized)
    axes[1].set_title("FPGA Input (32x32)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # --- Hardware Execution ---
    start_time = time.time()
    
    # Start IP and transfer data
    cnn_ip.write(0x00, 0x01)             
    dma.recvchannel.transfer(out_buffer) 
    dma.sendchannel.transfer(in_buffer)  
    
    # Wait for completion
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    
    # --- Post-Processing ---
    logits = np.copy(out_buffer)
    probs = softmax(logits)
    
    # Get top 5 predictions
    top5 = np.argsort(probs)[::-1][:5]
    top_idx = top5[0]
    top_label = classes[top_idx].upper()
    top_score = probs[top_idx] * 100

    # Print results
    print(f"⏱️ Inference Time: {inference_time_ms:.2f} ms")
    print("\n-------------------------------------------")
    print("📊 TOP 5 PREDICTIONS (FPGA HARDWARE):")
    print("-------------------------------------------")
    for i, idx in enumerate(top5):
        print(f"{i+1}. {classes[idx].upper():<10} : {probs[idx]*100:>6.2f}%")
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

    plt.title("FPGA Inference Result", fontsize=15)
    plt.axis('off')
    plt.show()

# ==========================================
# 3. RUN INFERENCE
# ==========================================
run_fpga_inference('/home/xilinx/pynq/overlays/final/test_image_9.jpg')
