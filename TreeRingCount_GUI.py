import math
import cv2
import numpy as np
from skimage import measure, morphology
from scipy import ndimage as ndi
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)


def custom_convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    # Determine the amount of padding needed
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Initialize the output image
    output_image = np.zeros_like(image)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and sum the result
            output_image[i, j] = np.sum(region * kernel)

    return output_image


def gaussian_blur(image, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    return custom_convolve2d(image, kernel)


def convolve_with_padding(image, kernel):
    padding_height = (kernel.shape[0] - 1) // 2
    padding_width = (kernel.shape[1] - 1) // 2

    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant',
                          constant_values=0)
    result = custom_convolve2d(padded_image, kernel)
    return result


def normalize_image(image):
    norm_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return norm_image.astype(np.uint8)


def median_filter(image, kernel_size=3):
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)),
                          mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.median(window)

    return filtered_image


class TreeAgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tree Age calculation")

        self.image_path = None
        self.processed_images = []
        self.current_image_index = 0
        self.captions = []

        self.create_widgets()


    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack()

        self.load_button = tk.Button(frame, text="Load Image", command=self.load_image, font=("Arial", 14), bg='#4CAF50', fg='white', padx=10, pady=5)
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.process_button = tk.Button(frame, text="Process Image", command=self.process_image, font=("Arial", 14), bg='#2196F3', fg='white', padx=10, pady=5)
        self.process_button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.prev_button = tk.Button(frame, text="Previous", command=self.prev_image, font=("Arial", 14), bg='#FFC107', fg='black', padx=10, pady=5)
        self.prev_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        self.next_button = tk.Button(frame, text="Next", command=self.next_image, font=("Arial", 14), bg='#FFC107', fg='black', padx=10, pady=5)
        self.next_button.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

        self.age_label = tk.Label(frame, text="Tree Age: ", font=("Helvetica", 22, "bold"))
        self.age_label.grid(row=3, column=1, padx=5, pady=5, sticky="nsew")

        self.fig_input = Figure(figsize=(4, 4))
        self.ax_input = self.fig_input.add_subplot(111)
        self.canvas_input = FigureCanvasTkAgg(self.fig_input, master=self.root)
        self.canvas_input.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_steps = Figure(figsize=(4, 4))
        self.ax_steps = self.fig_steps.add_subplot(111)
        self.canvas_steps = FigureCanvasTkAgg(self.fig_steps, master=self.root)
        self.canvas_steps.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_output = Figure(figsize=(4, 4))
        self.ax_output = self.fig_output.add_subplot(111)
        self.canvas_output = FigureCanvasTkAgg(self.fig_output, master=self.root)
        self.canvas_output.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.middle_frame = tk.Frame(self.root)
        self.middle_frame.pack()

        self.middle_age_label = tk.Label(self.middle_frame, text="", font=("Helvetica", 20, "bold"))
        self.middle_age_label.pack(pady=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.ax_input.clear()
            self.ax_input.imshow(image_rgb)
            self.ax_input.set_title("Original Image")
            self.canvas_input.draw()

    def process_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image loaded")
            return

        image = cv2.imread(self.image_path)
        processed_steps, captions, tree_age = self.segment_and_process_tree(image)

        self.processed_images = processed_steps
        self.captions = captions
        self.current_image_index = 0

        self.display_images()
        self.age_label.config(text=f"Tree Age: {tree_age} years")

    def next_image(self):
        if self.current_image_index < len(self.processed_images) - 1:
            self.current_image_index += 1
            self.display_images()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_images()

    def display_images(self):
        input_image = self.processed_images[0]
        current_step_image = self.processed_images[self.current_image_index]
        final_image = self.processed_images[-1]

        self.ax_input.clear()
        self.ax_input.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        self.ax_input.set_title("Original Image")
        self.canvas_input.draw()

        self.ax_steps.clear()
        self.ax_steps.imshow(cv2.cvtColor(current_step_image, cv2.COLOR_BGR2RGB))
        self.ax_steps.set_title(self.captions[self.current_image_index])
        self.canvas_steps.draw()

        self.ax_output.clear()
        self.ax_output.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        self.ax_output.set_title("Final Image with Lines")
        self.canvas_output.draw()

    def segment_and_process_tree(self, image):
        processed_steps = []
        captions = []

        # Segment the tree
        segmented_image = self.segment_tree(image)
        processed_steps.append(segmented_image)
        captions.append("Segmented Image")

        # Apply local histogram equalization
        local_equalized_image = self.apply_local_histogram_equalization(segmented_image)
        processed_steps.append(local_equalized_image)
        captions.append("Local Histogram Equalized Image")

        # Apply gamma correction
        gamma_corrected_image = self.apply_gamma_correction(local_equalized_image, gamma=1.5)
        processed_steps.append(gamma_corrected_image)
        captions.append("Gamma Corrected Image")

        # Convert to grayscale
        gray_image = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        processed_steps.append(thresh)
        captions.append("Adaptive Thresholding")

        # Morphological operations
        dilated_image, eroded_image = self.remove_noise_opening(thresh)
        processed_steps.append(eroded_image)
        captions.append("Eroded Image")
        processed_steps.append(dilated_image)
        captions.append("Dilated Image")

        img_median = self.remove_noise(dilated_image)
        processed_steps.append(img_median)
        captions.append("Median Filtered Image")

        # Thinning
        skeleton = self.zhang_suen_thinning(img_median)
        skeleton = self.closing(skeleton, 3)
        skeleton = self.remove_small_components(skeleton)
        processed_steps.append(skeleton)
        captions.append("Skeletonization")

        # Mark tree center
        marked_image, tree_center = self.find_and_mark_tree_center(skeleton)
        processed_steps.append(marked_image)
        captions.append(f"Tree Center: {tree_center}")

        # Add final image with lines and annotations
        ringcount_skeleton = skeleton
        top_most, bottom_most, left_most, right_most = self.find_extreme_points(skeleton)
        middle_vertical, middle_horizontal = tree_center

        avg_vertical_transitions, avg_horizontal_transitions, vertical_transitions_list, horizontal_transitions_list = self.count_transitions(ringcount_skeleton, middle_vertical, middle_horizontal)
        age = math.ceil((avg_vertical_transitions + avg_horizontal_transitions) / 2)

        # Draw lines on the final image
        final_image = cv2.cvtColor(ringcount_skeleton, cv2.COLOR_GRAY2BGR)
        for offset, color in zip([0, 10, -10], [(0, 0, 255), (255, 0, 0), (255, 0, 0)]):
            col = middle_horizontal + offset
            if 0 <= col < final_image.shape[1]:
                cv2.line(final_image, (col, 0), (col, final_image.shape[0]), color, 1)

        for offset, color in zip([0, 10, -10], [(0, 255, 0), (128, 0, 128), (128, 0, 128)]):
            row = middle_vertical + offset
            if 0 <= row < final_image.shape[0]:
                cv2.line(final_image, (0, row), (final_image.shape[1], row), color, 1)

        # Add average transition annotations
        avg_vertical_text = f'Vertical Transitions: {avg_vertical_transitions:.2f}'
        avg_horizontal_text = f'Horizontal Transitions: {avg_horizontal_transitions:.2f}'
        cv2.putText(final_image, avg_vertical_text, (10, final_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(final_image, avg_horizontal_text, (10, final_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        processed_steps.append(final_image)
        captions.append("Final Image with Lines")

        return processed_steps, captions, age

    def find_and_mark_tree_center(self, skeleton):
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

    def segment_tree(self, image):
        # Segment the tree from the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = binary_image > 0
        filled_image = ndi.binary_fill_holes(binary_image)
        labeled_image, _ = ndi.label(filled_image)
        regions = measure.regionprops(labeled_image)
        tree_region = max(regions, key=lambda x: x.area)
        tree_mask = labeled_image == tree_region.label
        tree_only = np.where(tree_mask[:, :, None], image_rgb, 255)
        return cv2.cvtColor(tree_only.astype('uint8'), cv2.COLOR_RGB2BGR)

    def apply_local_histogram_equalization(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_gamma_correction(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def remove_noise(self, image):
        image = cv2.medianBlur(image, 3)
        image = cv2.medianBlur(image, 5)
        return cv2.bilateralFilter(image, 100, 2, 2)

    def remove_noise_opening(self, image):
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        eroded_image = cv2.erode(binary_image, kernel, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        return dilated_image, eroded_image

    def closing(self, img, kernel_size=3):
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
        return cv2.erode(dilated_image, kernel, iterations=2)

    def zhang_suen_thinning(self, image):
        binary_image = (image // 255).astype(np.uint8)
        skeleton = morphology.thin(binary_image)
        return (skeleton * 255).astype(np.uint8)

    def remove_small_components(self, img, min_size=50):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                img[labels == i] = 0
        return img

    def find_extreme_points(self, binary_image):
        white_points = np.column_stack(np.where(binary_image == 255))
        top_most = white_points[white_points[:, 0].argmin()]
        bottom_most = white_points[white_points[:, 0].argmax()]
        left_most = white_points[white_points[:, 1].argmin()]
        right_most = white_points[white_points[:, 1].argmax()]
        return top_most, bottom_most, left_most, right_most

    def count_transitions(self, binary_image, middle_vertical, middle_horizontal):
        def transitions(line):
            return np.count_nonzero(line[:-1] != line[1:])
        vertical_transitions_list = [transitions(binary_image[:, col]) for col in [middle_horizontal, middle_horizontal + 10, middle_horizontal - 10] if 0 <= col < binary_image.shape[1]]
        horizontal_transitions_list = [transitions(binary_image[row, :]) for row in [middle_vertical, middle_vertical + 10, middle_vertical - 10] if 0 <= row < binary_image.shape[0]]
        avg_vertical_transitions = np.mean(vertical_transitions_list) / 4
        avg_horizontal_transitions = np.mean(horizontal_transitions_list) / 4
        return avg_vertical_transitions, avg_horizontal_transitions, vertical_transitions_list, horizontal_transitions_list

if __name__ == "__main__":
    root = tk.Tk()
    app = TreeAgeApp(root)
    root.mainloop()
