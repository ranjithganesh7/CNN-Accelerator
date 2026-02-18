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

## Demo
▶ Demo Video: 

## Repository Structure
See `/docs` for architecture and performance analysis.

## Authors
- Royce Niran George A
- Kamalesh S
- Ranjith Ganesh B

