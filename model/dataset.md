
# CIFAR-10 Dataset

## Overview
The CIFAR-10 (Canadian Institute For Advanced Research) dataset is a standard benchmark in computer vision and machine learning. It is widely used for training and evaluating small-scale Convolutional Neural Networks (CNNs) for image classification tasks.



## Dataset Specifications
* **Total Images:** 60,000
* **Image Dimensions:** 32x32 pixels
* **Color Space:** RGB (3 channels)
* **Number of Classes:** 10 (mutually exclusive)
* **Data Split:** * **Training Set:** 50,000 images (5,000 per class)
  * **Testing Set:** 10,000 images (1,000 per class)

## Class Categories
The dataset consists of the following 10 classes. The classes are completely mutually exclusive (e.g., "automobiles" includes sedans, SUVs, etc., while "trucks" includes only big trucks. Neither includes pickup trucks).

| Label Index | Class Name | Label Index | Class Name |
| :---: | :--- | :---: | :--- |
| **0** | Airplane | **5** | Dog |
| **1** | Automobile | **6** | Frog |
| **2** | Bird | **7** | Horse |
| **3** | Cat | **8** | Ship |
| **4** | Deer | **9** | Truck |

## Getting Started

Because CIFAR-10 is an industry standard, it is baked into the standard libraries of most modern deep learning frameworks. You don't need to manually download or parse the raw binaries.

### Loading in PyTorch
```python
import torchvision
import torchvision.transforms as transforms

# Define transformations (e.g., convert to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
