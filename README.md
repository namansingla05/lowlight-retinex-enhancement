# Low-Light Image Enhancement using Retinex with Adaptive Gamma and Dark Region Denoising

This project implements a **low-light image enhancement pipeline** using:
- **Iterative Least Squares (Retinex-based)**
- **Adaptive Gamma Correction**
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- **Color Preservation & Noise Reduction**
- **Image Quality Metrics** (Average Brightness, Discrete Entropy, NIQE)

It is designed to improve the **visibility, contrast, and quality** of low-light images while preserving natural colors and reducing noise.

---

## Features
- Enhances low-light images using Retinex-based illumination correction.
- Adaptive gamma correction for optimal brightness.
- CLAHE for local contrast enhancement.
- Preserves original colors to avoid unnatural tones.
- Optional noise reduction for dark areas.
- Computes quality metrics:
  - **AB (Average Brightness)**
  - **DE (Discrete Entropy)**
  - **NIQE (Natural Image Quality Evaluator)**

---

## Results — Input vs Output

Below is a collage showing the performance of the **Low-Light Image Enhancement** algorithm.  
The **top row** contains the original **low-light input images**, and the **bottom row** contains the corresponding **enhanced output images**.

![Low-Light Enhancement Comparison]([comparison.png](https://github.com/namansingla05/lowlight-retinex-enhancement/blob/main/Screenshot%202025-08-09%20221049.png))
