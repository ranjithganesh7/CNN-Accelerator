# Real-Time Object Detection Using Hardware-Accelerated CNN on Xilinx Zynq FPGA with Arm Processor

## Overview
This project demonstrates real-time Convolutional Neural Network (CNN) inference acceleration on a Xilinx Zynq-7000 SoC. By heavily leveraging hardware/software co-design, compute-intensive layers of the CNN are offloaded to the FPGA programmable logic (PL), while the Arm Cortex-A9 processor (PS) handles system control, data movement, and image preprocessing.

Developed as part of the Bharat AI-SoC Student Challenge and ARM hackathon, this hardware-accelerated CNN IP was built using Vitis HLS, integrated via Vivado, and deployed on a PYNQ-Z2 board.

## Key Features
- FPGA-Accelerated Inference: Custom IP for spatial convolution and pooling.
- HW/SW Co-Design: Seamless partitioning using Vitis HLS and Vivado.
- Real-Time Processing: High-throughput object detection and image classification.
- Quantitative Benchmarking: Built-in scripts to compare latency, throughput, and power against a purely software-driven CPU baseline.

## System Stack
- Hardware: Xilinx Zynq-7000 SoC (PYNQ-Z2 Development Board)
- Software Design: Vitis HLS 2023.1, Vivado Design Suite 2023.1
- Embedded Environment: PYNQ Linux, Python 3.x, OpenCV

## Performance Highlights
| Metric | CPU Only | FPGA Accelerated |
|------|---------|-----------------|
| Latency (ms) | 3347.1 | 114.117 |
| FPS | 1.86 | 8.76 |
| Throughput | 0.056 | 0.108 |
| Power Efficiency | Baseline | Improved |

## GitHub Repository Structure
```
fpga-cnn-accelerator-zynq/
│
├── README.md
├── LICENSE
├── docs/
│   ├── architecture.md
│   ├── design_partitioning.md
│   ├── performance_analysis.md
│   ├── cpu_vs_fpga_comparison.md
│   ├── power_analysis.md
│   ├── resource_utilization.md
│   └── diagrams/
│       ├── system_architecture.png
│       ├── hw_sw_partition.png
│       ├── cnn_accelerator_pipeline.png
│
├── hls/
│   ├── cnn_accelerator.cpp
│   ├── cnn_accelerator.h
│   ├── tb_cnn.cpp
│   ├── vitis_hls_project/
│
├── vivado/
│   ├── block_design/
│   ├── constraints/
│   ├── bitstream/
│   │   └── design.bit
│
├── software/
│   ├── cpu_only/
│   │   ├── cnn_cpu.cpp
│   │   └── benchmark_cpu.py
│   ├── fpga_accel/
│   │   ├── overlay.bit
│   │   ├── overlay.hwh
│   │   ├── inference_fpga.py
│   │   └── dma_utils.py
│
├── dataset/
│   └── sample_images/
│
├── results/
│   ├── latency_fps.csv
│   ├── accuracy_results.csv
│   └── screenshots/
│
├── demo/
│   ├── demo_video_link.txt
│   └── demo_script.md
│
└── report/
    ├── Final_Project_Report.pdf
    └── figures/
```

## Demo
▶ Demo Video: [https://drive.google.com/file/d/1zEg1VLj9V9LxN7ka2FiaaFTp8HAQc8l4/view?usp=drive_link](https://drive.google.com/file/d/1zEg1VLj9V9LxN7ka2FiaaFTp8HAQc8l4/view?usp=drive_link)


## 🚀 How to Run

### Prerequisites
- PYNQ-Z2 (or compatible Zynq board)
- SD Card (≥16GB recommended)
- PYNQ Linux image
- Python 3.x
- OpenCV installed on board
- FPGA bitstream (`.bit`) and hardware handoff file (`.hwh`)
- CNN model weights

### Hardware Setup
1. Flash the PYNQ image to the SD card  
2. Insert the SD card into the board  
3. Connect power, Ethernet/USB, and camera (optional)  
4. Power ON the board  

### Deploy FPGA Bitstream
```bash
scp overlay.bit xilinx@<board_ip>:/home/xilinx/
scp overlay.hwh xilinx@<board_ip>:/home/xilinx/
python3 load_overlay.py
```

## ▶️ Execution Commands

### Run CPU-Only Baseline
```bash
cd software/cpu_only
python3 cnn_cpu.py --image sample.jpg
```

### CPU-Only Baseline
```bash
cd software/cpu_only
python3 cnn_cpu.py --image sample.jpg
```
Output: Prediction, confidence, latency (ms), FPS

### FPGA-Accelerated Inference
```bash
cd ../fpga_accel
python3 inference_fpga.py --image sample.jpg
```
Output: Prediction, confidence, latency (ms), FPS

### Live Camera Inference
```bash
python3 inference_fpga.py --camera 1
```
Press q to exit

### Performance Benchmarking
```bash
python3 benchmark.py
```
Output: Average latency, FPS, speedup ratio

## Repository Structure
See `/docs` for architecture and performance analysis.

## Authors
- Royce Niran George A
- Kamalesh S
- Ranjith Ganesh B

