# Real-Time Object Detection Using Hardware-Accelerated CNN on Xilinx Zynq FPGA with Arm Processor

## Overview
This project demonstrates real-time CNN inference acceleration on a Xilinx Zynq SoC by offloading compute-intensive layers to FPGA fabric while using the Arm processor for control and preprocessing.

A hardware-accelerated CNN IP was developed using Vitis HLS, integrated into Vivado, and deployed on a PYNQ-Z2 board.

## Key Features
- FPGA-accelerated CNN inference
- Hardware/Software co-design using Vitis and Vivado
- Real-time image classification/object detection
- Quantitative comparison with CPU-only implementation

## Hardware Platform
- Xilinx Zynq SoC (PYNQ-Z2)
- ARM Cortex-A9
- FPGA Fabric with DSP acceleration

## Software Stack
- Vitis HLS
- Vivado Design Suite
- PYNQ Framework
- Python + OpenCV

## Performance Highlights
| Metric | CPU Only | FPGA Accelerated |
|------|---------|-----------------|
| Latency (ms) | XX | XX |
| FPS | XX | XX |
| Speedup | XX | XX |
| Power Efficiency | XX | XX |

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
▶ Demo Video: 


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

