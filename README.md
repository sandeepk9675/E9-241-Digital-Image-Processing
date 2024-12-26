# E9-241 Digital Image Processing Assignments

This repository contains the assignments for the E9-241 Digital Image Processing course. The assignments focus on implementing and understanding various image processing techniques, including histograms, binarization, spatial filtering, frequency domain analysis, connected components, and deep learning-based image classification.

---

## Assignment 1: Histogram and Binarization Techniques

1. **Histogram Computation**:
   - Compute and plot the histogram of `coins.png`.
   - Calculate the average intensity from the histogram and verify with the actual average intensity.

2. **Otsu’s Binarization**:
   - Implement binarization by minimizing within-class variance and maximizing between-class variance.
   - Plot and verify equivalence of the methods.

3. **Adaptive Binarization**:
   - Divide `sudoku.png` into N×N blocks and apply Otsu’s binarization.
   - Compare results for block sizes: 2×2, 4×4, 8×8, and 16×16.

4. **Connected Components**:
   - Binarize `quote.png` and count characters excluding punctuation using 8-neighbor connectivity.
   - Ensure vectorized implementation.

---

## Assignment 2: Spatial Filtering and Scaling

1. **Gaussian Blurring and Binarization**:
   - Apply Gaussian blur to `moon_noisy.png` using a 41×41 filter.
   - Perform Otsu’s binarization for different σ values and find the optimal σ.

2. **Fractional Scaling with Interpolation**:
   - Downsample `flowers.png` by 2 and upsample by 3 using bilinear interpolation.
   - Compare results with upsampling by 1.5.

3. **Photoshop Brightness/Contrast Feature**:
   - Implement `brightnessAdjust(img, p)` and `contrastAdjust(img, p)`.
   - Test with `brightness_contrast.jpg`.

---

## Assignment 3: Frequency Domain Filtering

1. **Ideal Filters**:
   - Apply ILPF, IBPF, and IHPF to `dynamicCheckerBoard.png` with specified parameters.
   - Observe and analyze results.

2. **Gaussian Filtering**:
   - Apply ILPF and GLPF to `characters.tif` with D0 = 100.
   - Compare results for artifacts.

---

## Assignment 4: Image Denoising and Hough Transform

1. **Denoising**:
   - Use bilateral filtering on `building_noisy.png` and compare with Gaussian smoothing.
   - Apply Laplace filter to analyze results.

2. **Hough Transform**:
   - Detect circles in images with varying conditions (noise, radius mismatch, occlusion).
   - Analyze effects of different edge detectors.

---

## Assignment 5: Deep Features for Image Classification

1. **Feature Extraction**:
   - Use a pre-trained model to extract deep features for classification using KNN (k=3).

2. **Noise Analysis**:
   - Add noise to train/test images and evaluate classification accuracy.

3. **Fine-Tuning**:
   - Fine-tune the last layer and compare results with previous experiments.
---

### Submission:
- Submit code, plots, and a report summarizing the implementation and results.


### Repository Structure:
- `Assignment1/`: Histogram and binarization techniques.
- `Assignment2/`: Spatial filtering and scaling.
- `Assignment3/`: Frequency domain filtering.
- `Assignment4/`: Denoising and Hough transform.
- `Assignment5/`: Deep features for classification.

---
