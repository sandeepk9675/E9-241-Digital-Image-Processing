# Numerical Linear Algebra Assignments

This repository contains solutions for three Numerical Linear Algebra assignments, showcasing the application of advanced computational techniques to solve real-world problems.

---

# E9-241 Digital Image Processing Assignments

This repository contains solutions for the assignments of the E9-241 Digital Image Processing course. These solutions demonstrate various image processing techniques, including histograms, binarization, spatial filtering, frequency domain analysis, connected components, and deep learning-based image classification.

---

## Assignment 1: Histogram and Binarization Techniques

1. **Histogram Computation**:
   - Computed and plotted the histogram of `coins.png`.
   - Calculated the average intensity from the histogram and verified it with the actual average intensity.

2. **Otsu’s Binarization**:
   - Implemented binarization by minimizing within-class variance and maximizing between-class variance.
   - Plotted and verified the equivalence of the methods.

3. **Adaptive Binarization**:
   - Divided `sudoku.png` into N×N blocks and applied Otsu’s binarization.
   - Compared results for block sizes: 2×2, 4×4, 8×8, and 16×16.

4. **Connected Components**:
   - Binarized `quote.png` and counted characters excluding punctuation using 8-neighbor connectivity.
   - Ensured a vectorized implementation.

---

## Assignment 2: Spatial Filtering and Scaling

1. **Gaussian Blurring and Binarization**:
   - Applied Gaussian blur to `moon_noisy.png` using a 41×41 filter.
   - Performed Otsu’s binarization for different σ values and identified the optimal σ.

2. **Fractional Scaling with Interpolation**:
   - Downsampled `flowers.png` by 2 and upsampled by 3 using bilinear interpolation.
   - Compared results with upsampling by 1.5.

3. **Photoshop Brightness/Contrast Feature**:
   - Implemented `brightnessAdjust(img, p)` and `contrastAdjust(img, p)`.
   - Tested with `brightness_contrast.jpg`.

---

## Assignment 3: Frequency Domain Filtering

1. **Ideal Filters**:
   - Applied ILPF, IBPF, and IHPF to `dynamicCheckerBoard.png` with specified parameters.
   - Observed and analyzed results.

2. **Gaussian Filtering**:
   - Applied ILPF and GLPF to `characters.tif` with D0 = 100.
   - Compared results for artifacts.

---

## Assignment 4: Image Denoising and Hough Transform

1. **Denoising**:
   - Used bilateral filtering on `building_noisy.png` and compared it with Gaussian smoothing.
   - Applied Laplace filter to analyze results.

2. **Hough Transform**:
   - Detected circles in images with varying conditions (noise, radius mismatch, occlusion).
   - Analyzed effects of different edge detectors.

---

## Assignment 5: Deep Features for Image Classification

1. **Feature Extraction**:
   - Used a pre-trained model to extract deep features for classification using KNN (k=3).

2. **Noise Analysis**:
   - Added noise to train/test images and evaluated classification accuracy.

3. **Fine-Tuning**:
   - Fine-tuned the last layer and compared results with previous experiments.

---

### Repository Structure:
- `Assignment1/`: Solutions for histogram and binarization techniques.
- `Assignment2/`: Solutions for spatial filtering and scaling.
- `Assignment3/`: Solutions for frequency domain filtering.
- `Assignment4/`: Solutions for denoising and Hough transform.
- `Assignment5/`: Solutions for deep features and classification.

---

This repository showcases implemented solutions, complete with code, plots, and results. Feedback and suggestions are welcome!

