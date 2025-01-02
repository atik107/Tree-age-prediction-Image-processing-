# Tree Age Determination Using Ring Count

A project focused on estimating tree age using image processing techniques by analyzing tree ring patterns. This project is part of the CSE 4120: Image Processing & Computer Vision Laboratory course at Khulna University of Engineering & Technology.

---

## Table of Contents
- [Introduction](#introduction)
- [Applications](#applications)
- [Key Steps](#key-steps)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction
Tree ring analysis, or dendrochronology, is a scientific method of dating based on the patterns of tree rings. This project utilizes image processing techniques to determine tree age from cross-section images of tree trunks.

---

## Applications
- Forest management
- Tree growth rate analysis
- Ecology and biodiversity studies
- Wood quality assessment

---

## Key Steps
1. Image preprocessing
2. Segmentation
3. Local histogram equalization
4. Gamma correction
5. Noise removal (Gaussian Blur, Median Filter)
6. Adaptive thresholding
7. Morphological operations (Opening, Skeletonization, Closing)
8. Tree ring count estimation

---

## Methodology
### Image Segmentation
- Convert RGB image to grayscale
- Apply Otsu's thresholding
- Fill holes in the binary image
- Label and isolate the largest region representing the tree cross-section

### Local Histogram Equalization
- Enhance contrast in poorly visible regions using CLAHE.

### Gamma Correction
- Adjust brightness and contrast using a gamma value of 1.5.

### Adaptive Thresholding
- Automatically binarize the image based on local pixel intensity variations.

### Morphological Operations
- Apply opening, closing, and thinning to refine tree ring structures.

---

## Results
- GUI for loading images and performing all processing steps.
- Estimation of tree age by averaging transitions in the skeletonized image.

---

## Limitations
- Misidentification of rings due to poor contrast or overlapping growth rings.
- Challenges with non-circular or irregular cross-sections.
- Difficulty in detecting rings with very narrow widths.

---

## Conclusion
- Effective segmentation and tree ring isolation.
- Improved detection with advanced image processing techniques.
- Potential for large-scale automation in dendrochronology.

---

## References
1. P. M. Sundari, et al., "An Approach for Analyzing the Factors Recorded in the Tree Rings Using Image Processing Techniques," 2017.
2. S. S. P. Gannamani, et al., "Tree Age Predictor Using Augmented Reality and Image Processing Techniques," 2021.
3. Schweingruber, F.H., *Tree rings: basics and applications of dendrochronology*, Springer, 2012.
4. Baillie, M.G., *Tree-ring dating and archaeology*, Routledge, 2014.

---

## Acknowledgments
Supervised by:  
- Dr. Sk. Md. Masudul Ahsan, Professor, Department of Computer Science and Engineering  
- Dipannita Biswas, Lecturer, Department of Computer Science and Engineering  

Presented by:  
Atiqul Islam Atik  
Roll: 1907107  
Khulna University of Engineering & Technology
