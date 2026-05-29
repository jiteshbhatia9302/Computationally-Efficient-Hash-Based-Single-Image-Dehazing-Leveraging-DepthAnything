# Computationally Efficient Hash Based Single Image Dehazing Leveraging DepthAnything

This project implements a complete single image dehazing pipeline using:

- Depth estimation using DepthAnything
- Atmospheric Light estimation using Channel Attention
- Transmission map estimation using depth information
- Atmospheric Scattering Model (ASM)
- Post-processing enhancement in HSI color space

---

# Project Workflow

The pipeline performs the following steps:

1. Read input hazy image
2. Estimate depth map using DepthAnything
3. Estimate atmospheric light using channel attention
4. Estimate transmission map using depth information
5. Compute concatenated vector G(x)
6. Recover haze-free image using ASM
7. Enhance image using histogram equalization in HSI space

---

# Folder Structure

```text
project/
│
├── main.py
├── image_reader.py
├── depth_estimation.py
├── atmospheric_light.py
├── transmission_estimation.py
├── concatenated_vector.py
├── asm_model.py
├── color_space.py
├── histogram_utils.py
├── post_processing.py
├── requirements.txt
├── README.md
├── input.jpg
```

---

# Installation

## 1. Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# PyTorch CUDA Support (Optional)

For NVIDIA GPU support, install CUDA-enabled PyTorch.

Visit:

https://pytorch.org/get-started/locally/

Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

# Input Image

Place the hazy image inside the project folder and rename it as:

```text
input.jpg
```

---

# Running the Project

Execute:

```bash
python main.py
```

---

# Output Files

After execution, the following files are generated:

```text
final_output.jpg
depth_map.jpg
transmission_map.jpg
```

---

# Algorithm Details

## 1. Depth Estimation

DepthAnything is used to estimate the monocular depth map of the hazy image.

---

## 2. Atmospheric Light Estimation

Atmospheric light is estimated using Channel Attention:

\[
Z = PWConv(GELU(PWConv(GAP(f(x)))))
\]

\[
CA_c = sigmoid(Z)
\]

\[
A = f(x) \odot CA_c
\]

---

## 3. Transmission Map

Transmission is estimated using:

\[
t(x) = e^{-\beta d(x)}
\]

SHA-256 hashing is used to cache transmission values corresponding to identical depth values.

---

## 4. Concatenated Vector

\[
G(x) =
\frac{
(I(x)-A)/t(x) + (A-C)
}{
I(x)-1
}
\]

---

## 5. Atmospheric Scattering Model

\[
J(x) = G(x)(I(x)-1) + C
\]

---

## 6. Post Processing

The reconstructed image is enhanced using:

- RGB-to-HSI conversion
- Histogram Equalization
- Contrast normalization
- HSI-to-RGB reconstruction

---

# Recommended Python Version

```text
Python 3.9 – 3.11
```

---

# Author

Implemented for research purposes.