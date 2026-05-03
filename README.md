## Project Overview

This project tackles the problem of translating a **digital clock representation** into a corresponding **analog clock image** using deep learning.

Given:

* A digital clock image (time input)
* An analog clock template

The goal is to generate an analog clock that accurately reflects the given digital time.

---

## Problem Definition

This is a **supervised learning problem** where the model learns a mapping:

[
f(x; W): \text{Digital Representation} \rightarrow \text{Analog Clock Image}
]

The challenge involves:

* Understanding **time representation**
* Learning **clock geometry**
* Generating **visually coherent images**

---

## Methodology

Instead of solving the task end-to-end, the problem is decomposed into three sub-tasks:

### geometry Extraction Model

* Learns the **structure of an analog clock**
* Focuses on:

  * Clock face
  * Numbers / markers
  * Circular layout
* Result: A clean and consistent clock template

---

### Eraser Model (Background Reconstruction)

* Removes clock hands from an image
* Learns to separate:

  * **Foreground (hands)**
  * **Background (clock face)**
* Result: Reconstructed clean background

Limitation:

* Fine details (numbers, thin lines) may be blurred due to MSE loss

---

###  Time Mapping Model (ResNet Regression)

* Maps digital time → analog representation
* Learns:

  * Angles of hour and minute hands
* Implemented using a **ResNet-based architecture**

Limitation:

* Some degree of **overfitting observed**

---

##  Full Pipeline

1. Generate / clean clock geometry
2. Remove existing hands (Eraser model)
3. Predict correct hand positions (Time Mapping)
4. Combine outputs → Final analog clock

---

## Models & Techniques

* **CNNs** for spatial feature extraction
* **ResNet** for regression (time → angles)
* **MSE Loss** for optimization
* **Gradient Descent + Backpropagation** for training

---

## Training Insights

###  Geometry Model

* Stable training
* Good generalization
* Low gap between train and validation loss

###  Eraser Model

* Successfully reconstructs background
* Loses high-frequency details (blur effect)

###  Time Mapping Model

* Shows moderate overfitting
* Validation loss plateaus after certain epochs

---

##  Challenges

* Mapping abstract time → spatial geometry
* Preserving fine image details
* Avoiding overfitting with limited data
* Multi-stage dependency between models

---

##Future Improvements

* Replace MSE with **L1 / perceptual loss**
* Use **U-Net** for better image reconstruction
* Add **data augmentation**
* Apply **early stopping** and regularization
* Explore **end-to-end architecture**

---

##  Repository Structure

```
├── data/                # Generated and processed datasets
├── models/              # Model definitions (CNN, ResNet)
├── training/            # Training scripts
├── utils/               # Helper functions
├── notebooks/           # Experiments and visualization
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Run inference
python inference.py
```

---

##  Key Takeaways

* Complex problems can benefit from **task decomposition**
* CNNs are effective for spatial reasoning tasks
* Loss function choice strongly affects visual quality
* Overfitting and blurring are common challenges in image-based models