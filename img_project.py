import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage as ndi
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def segment_tree(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Threshold the image using Otsu's method
    threshold_value, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill holes in the binary image
    binary_image = binary_image > threshold_value
    filled_image = ndi.binary_fill_holes(binary_image)

    # Label objects in the image
    labeled_image, _ = ndi.label(filled_image)
    regions = measure.regionprops(labeled_image)

    # Assume the largest region is the tree cross-section
    tree_region = max(regions, key=lambda x: x.area)
    tree_mask = labeled_image == tree_region.label

    # Apply the mask to the original image to isolate the tree
    tree_only = np.where(tree_mask[:, :, None], image_rgb, 255)  # Using 255 for the background to make it white

    # Convert the result to an image
    result_image = cv2.cvtColor(tree_only.astype('uint8'), cv2.COLOR_RGB2BGR)

    return result_image


def apply_local_histogram_equalization(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    equalized_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return equalized_image


def apply_gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def remove_noise(image):
    # Use a series of median filters to reduce noise
    image = cv2.medianBlur(image, 3)
    image = cv2.medianBlur(image, 5)
    # Apply bilateral filter for further noise reduction
    image = cv2.bilateralFilter(image, 100, 2, 2)  # Adjust parameters as needed
    return image


def remove_noise_opening(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    return dilated_image, eroded_image

def closing(img, kernel_size=3):
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
    closed_image = cv2.erode(dilated_image, kernel, iterations=2)
    return closed_image

def zhang_suen_thinning(image):
    # Ensure the image is in binary format (0 or 1)
    binary_image = (image // 255).astype(np.uint8)

    # Apply the Zhang-Suen thinning algorithm
    skeleton = morphology.thin(binary_image)

    # Convert the result back to 8-bit image (0 or 255)
    thinned_image = (skeleton * 255).astype(np.uint8)

    return thinned_image


def remove_small_components(img, min_size=50):
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Iterate through each component and remove small ones
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            img[labels == i] = 0
    return img





# Load the image
image_path = "Tree_images/tree_seg_1.png"


segmented_image = segment_tree(image_path)
local_equalized_image_show = apply_local_histogram_equalization(segmented_image)

segmented_image = remove_noise(segmented_image)
segmented_image = remove_noise(segmented_image)


# Apply local histogram equalization
local_equalized_image = segmented_image
for i in range(0,2):
    local_equalized_image = apply_local_histogram_equalization(local_equalized_image)


# cv2.imshow("pic",local_equalized_image)

# Apply gamma correction
gamma = 1.5
gamma_corrected_image_show = apply_gamma_correction(local_equalized_image_show, gamma)
gamma_corrected_image = apply_gamma_correction(local_equalized_image, gamma)

# Convert gamma_corrected_image to grayscale for further processing
segmented_image_gray = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(segmented_image_gray, (5, 5), 0)

# Use adaptive thresholding to highlight the rings
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


# Morphological opening for noise removal
dilated_image, eroded_image = remove_noise_opening(thresh)

# Median for noise removal
img_median = remove_noise(dilated_image)

# Thinning
skeleton = zhang_suen_thinning(img_median)

#Morphological closing
#skeleton = closing(skeleton,3)

# connected component analysis
skeleton = remove_small_components(skeleton)




# Find contours of the skeleton
contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = []
def draw_filtered_contours(image, contours, min_contour_area=100):
    # Filter out small and irrelevant contours
    global filtered_contours
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]

    # Convert the image to grayscale and then back to BGR
    segmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rings_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)

    # Define colors for drawing contours
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (128, 128, 0),  # Olive
        (0, 128, 0),  # Dark Green
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (192, 192, 192),  # Silver
        (255, 165, 0),  # Orange
        (255, 192, 203),  # Pink
        (75, 0, 130),  # Indigo
        (255, 105, 180),  # Hot Pink
        (173, 216, 230),  # Light Blue
        (0, 255, 127),  # Spring Green
        (255, 69, 0),  # Orange Red
        (250, 128, 114),  # Salmon
    ]

    # Draw contours (rings) on the image with different colors
    for i, contour in enumerate(filtered_contours):
        color = colors[i % len(colors)]
        cv2.drawContours(rings_image, [contour], -1, color, 2)

    return rings_image











rings_image = draw_filtered_contours(segmented_image, contours, 100)
# Count the number of rings
num_rings = len(filtered_contours)

plt.figure(figsize=(14, 16))

plt.subplot(3, 4, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

plt.subplot(3, 4, 2)
plt.title('Local Histogram Equalized Image')
plt.imshow(cv2.cvtColor(local_equalized_image_show, cv2.COLOR_BGR2RGB))

plt.subplot(3, 4, 3)
plt.title('Gamma Corrected Image')
plt.imshow(cv2.cvtColor(gamma_corrected_image_show, cv2.COLOR_BGR2RGB))

plt.subplot(3, 4, 4)
plt.title('Blurred Image')
plt.imshow(blurred, cmap='gray')

plt.subplot(3, 4, 5)
plt.title('Adaptive Thresholding')
plt.imshow(thresh, cmap='gray')

plt.subplot(3, 4, 6)
plt.title('Eroded image')
plt.imshow(eroded_image, cmap='gray')

plt.subplot(3, 4, 7)
plt.title('Dialated image')
plt.imshow(dilated_image, cmap='gray')

plt.subplot(3, 4, 8)
plt.title('After apply median')
plt.imshow(img_median, cmap='gray')



plt.subplot(3, 4, 10)
plt.title('Skeletonization')
plt.imshow(skeleton, cmap='gray')


plt.subplot(3, 4, 11)
plt.title(f'Detected Rings with Outer Ring (Count: {num_rings})')
plt.imshow(cv2.cvtColor(rings_image, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

print(f'Number of rings detected: {num_rings}')


def find_extreme_points(binary_image):
    # Find coordinates of all white points
    white_points = np.column_stack(np.where(binary_image == 255))

    # Find the top-most, bottom-most, left-most, and right-most points
    top_most = white_points[white_points[:, 0].argmin()]
    bottom_most = white_points[white_points[:, 0].argmax()]
    left_most = white_points[white_points[:, 1].argmin()]
    right_most = white_points[white_points[:, 1].argmax()]

    return top_most, bottom_most, left_most, right_most

def count_transitions(binary_image, middle_vertical, middle_horizontal):
    # Initialize lists to store transitions
    vertical_transitions_list = []
    horizontal_transitions_list = []

    # Function to count transitions for a given line
    def transitions(line):
        return np.count_nonzero(line[:-1] != line[1:])

    # Vertical lines: middle, +10 pixels, -10 pixels
    for offset in [0, 10, -10]:
        col = middle_horizontal + offset
        if 0 <= col < binary_image.shape[1]:
            vertical_line = binary_image[:, col]
            vertical_transitions_list.append(transitions(vertical_line))

    # Horizontal lines: middle, +10 pixels, -10 pixels
    for offset in [0, 10, -10]:
        row = middle_vertical + offset
        if 0 <= row < binary_image.shape[0]:
            horizontal_line = binary_image[row, :]
            horizontal_transitions_list.append(transitions(horizontal_line))

    # Calculate the average number of transitions
    avg_vertical_transitions = np.mean(vertical_transitions_list) / 4
    avg_horizontal_transitions = np.mean(horizontal_transitions_list) / 4

    return avg_vertical_transitions, avg_horizontal_transitions, vertical_transitions_list, horizontal_transitions_list


def find_and_mark_tree_center(skeleton):
    def transitions(line):
        return np.count_nonzero(line[:-1] != line[1:])

    # Initialize variables to store the maximum transition counts
    max_vertical_transitions = 0
    max_horizontal_transitions = 0
    best_vertical_line = 0
    best_horizontal_line = 0

    # Iterate over each column (vertical lines)
    for col in range(skeleton.shape[1]):
        vertical_line = skeleton[:, col]
        vertical_transitions = transitions(vertical_line)
        if vertical_transitions > max_vertical_transitions:
            max_vertical_transitions = vertical_transitions
            best_vertical_line = col

    # Iterate over each row (horizontal lines)
    for row in range(skeleton.shape[0]):
        horizontal_line = skeleton[row, :]
        horizontal_transitions = transitions(horizontal_line)
        if horizontal_transitions > max_horizontal_transitions:
            max_horizontal_transitions = horizontal_transitions
            best_horizontal_line = row

    # The intersection point of the best vertical and horizontal lines is the tree center
    tree_center = (best_horizontal_line, best_vertical_line)

    # Mark the tree center in red
    marked_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    cv2.circle(marked_image, (tree_center[1], tree_center[0]), 5, (0, 0, 255), -1)

    # Draw additional lines through the tree center for visualization
    for angle in range(0, 360, 60):
        x_offset = int(1000 * math.cos(math.radians(angle)))
        y_offset = int(1000 * math.sin(math.radians(angle)))
        cv2.line(marked_image, (tree_center[1], tree_center[0]), (tree_center[1] + x_offset, tree_center[0] + y_offset),
                 (0, 255, 0), 1)

    return marked_image, tree_center



# Find extreme points in the skeleton image
top_most, bottom_most, left_most, right_most = find_extreme_points(skeleton)
marked_image, tree_center = find_and_mark_tree_center(skeleton)

# Display the result
plt.imshow(marked_image)
plt.title(f'Tree Center: {tree_center}')
plt.show()

middle_vertical , middle_horizontal = tree_center

# Count transitions in the skeleton image
avg_vertical_transitions, avg_horizontal_transitions, vertical_transitions_list, horizontal_transitions_list = count_transitions(skeleton, middle_vertical, middle_horizontal)

# Display the transition counts
print(f'Average number of transitions along the vertical lines: {avg_vertical_transitions}')
print(f'Average number of transitions along the horizontal lines: {avg_horizontal_transitions}')

# Plotting the skeleton image with lines for visualization
plt.figure(figsize=(8, 8))
plt.title('Skeleton with Analysis Lines')
plt.imshow(skeleton, cmap='gray')

# Draw the vertical lines
for offset, color in zip([0, 10, -10], ['red', 'blue', 'blue']):
    col = middle_horizontal + offset
    if 0 <= col < skeleton.shape[1]:
        plt.axvline(x=col, color=color, linestyle='--')

# Draw the horizontal lines
for offset, color in zip([0, 10, -10], ['green', 'purple', 'purple']):
    row = middle_vertical + offset
    if 0 <= row < skeleton.shape[0]:
        plt.axhline(y=row, color=color, linestyle='--')

# Annotate transition counts on the image
plt.text(middle_horizontal + 10, skeleton.shape[0] - 10, f'Avg Vertical Transitions: {avg_vertical_transitions:.2f}', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.text(10, middle_vertical + 30, f'Avg Horizontal Transitions: {avg_horizontal_transitions:.2f}', color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))


age = (avg_vertical_transitions + avg_horizontal_transitions) / 2
age = math.ceil(age)

# Add text annotation below the image
plt.figtext(0.5, 0.01, f'Tree Age (years): {age}', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.show()


