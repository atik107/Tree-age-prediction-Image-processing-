import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#blur filter to smooth image
def generate_gaussian_kernel(sigmaX, sigmaY, MUL=7, center_x=-1, center_y=-1):
    w = int(sigmaX * MUL) | 1
    h = int(sigmaY * MUL) | 1

    center_x = w // 2 if center_x == -1 else center_x
    center_y = h // 2 if center_y == -1 else center_y

    kernel = np.zeros((w, h))
    c = 1 / (2 * math.pi * sigmaX * sigmaY)

    for x in range(w):
        for y in range(h):
            dx = x - center_x
            dy = y - center_y

            x_part = (dx * dx) / (sigmaX * sigmaX)
            y_part = (dy * dy) / (sigmaY * sigmaY)

            kernel[x][y] = c * math.exp(- 0.5 * (x_part + y_part))

    formatted_kernel = (kernel / np.min(kernel)).astype(int)

    return formatted_kernel



def generate_mean_kernel(rows=3, cols=3):
    kernel = np.zeros((rows, cols))

    for x in range(0, rows):
        for y in range(0, cols):
            kernel[x, y] = 1.0

    return kernel / (rows * cols)


def generate_laplacian_kernel(negCenter=True):
    n = 3
    other_val = 1
    if not negCenter:
        other_val = -1

    kernel = other_val * np.ones((n, n))
    center = n // 2
    kernel[center, center] = - other_val * (n * n - 1)

    # print(kernel)
    return kernel


def generate_log_kernel(sigma, MUL=7):
    n = int(sigma * MUL)
    n = n | 1
    kernel = np.zeros((n, n))
    center = n // 2
    part1 = -1 / (np.pi * sigma ** 4)

    for x in range(n):
        for y in range(n):
            dx = x - center
            dy = y - center
            part2 = (dx ** 2 + dy ** 2) / (2 * sigma ** 2)
            kernel[x][y] = part1 * (1 - part2) * np.exp(-part2)

    mn = np.min(np.abs(kernel))
    return kernel



def generate_sobel_kernel(horiz=True):
    if horiz:
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    else:
        kernel = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

    return kernel








#Image related Function

def normalize(image):
    cv2.normalize(image,image,0,255,cv2.NORM_MINMAX)
    return np.round(image).astype(np.uint8)


def pad_image(image, kernel_height, kernel_width, kernel_center):
    pad_top = kernel_center[0]
    pad_bottom = kernel_height - kernel_center[0] - 1
    pad_left = kernel_center[1]
    pad_right = kernel_width - kernel_center[1] - 1

    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_image


def convolve(image, kernel, kernel_center=(-1, -1)):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    padded_image = np.pad(image, ((kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype='float32')

    for y in range(kernel_height // 2, image_height + kernel_height // 2):
        for x in range(kernel_width // 2, image_width + kernel_width // 2):
            sum = 0
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    sum += kernel[ky, kx] * padded_image[y - kernel_height // 2 + ky, x - kernel_width // 2 + kx]
            output[y - kernel_height // 2, x - kernel_width // 2] = sum

    return output





#perform sobel kernel convolution

def perform_sobel(imagePath, kernel_center=(-1, -1)):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    kernel_horiz = generate_sobel_kernel()
    image_horiz = convolve(image=image, kernel=kernel_horiz, kernel_center=kernel_center)

    # image_horiz = normalize(image_horiz)
    cv2.imshow('Horizontal img', image_horiz)
    cv2.waitKey(0)

    kernel_vert = generate_sobel_kernel(horiz=False)
    image_vert = convolve(image=image, kernel=kernel_vert, kernel_center=kernel_center)
    # image_vert = normalize(image_vert)

    cv2.imshow('Vertical img', image_vert)
    cv2.waitKey(0)

    height, width = image.shape
    out = np.zeros_like(image, dtype='float32')

    for i in range(0, height):
        for j in range(0, width):
            dx = image_horiz[i,j]
            dy = image_vert[i,j]

            res = math.sqrt(dx ** 2 + dy ** 2)
            out[i,j]=res

    out = normalize(out)

    cv2.imshow('Output image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






#Image conversion
def extract_rgb(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    return (red_channel, green_channel, blue_channel)

def merge_rgb( red, green, blue ):
    return cv2.merge( [blue, green, red] )

def extract_hsv(image):
    h_channel, s_channel, v_channel = cv2.split(image)
    return (h_channel, s_channel, v_channel)

def merge_hsv(h, s, v):
    return cv2.merge([h, s, v])

def hsv_to_rgb(hsv):
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def rgb_to_hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)





# Convolution on hsv
def convolution_hsv(image, kernel, kernel_center=(-1, -1)):
    image = rgb_to_hsv(image)
    hue, sat, val = extract_hsv(image=image)

    # Convolve each channel
    hue_normalization = normalize(convolve(hue, kernel, kernel_center))
    sat_normalization = normalize(convolve(sat, kernel, kernel_center))
    val_normalization = normalize(convolve(val, kernel, kernel_center))

    #cv2.imshow("Extracted Hue", hue_normalization)
    #cv2.imshow("Extracted Sat", sat_normalization)
    #cv2.imshow("Extracted Val", val_normalization)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    merged_hsv = merge_hsv(h=hue_normalization, s=sat_normalization, v=val_normalization)
    # merged_rgb = hsv_to_rgb(merge_hsv)

    orignial_rgb = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    merged_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Original(HSV) image", image)
    cv2.imshow("Merged(HSV) image", merged_hsv)
    cv2.waitKey(0)

    #cv2.destroyAllWindows()
    return merged_hsv



# Convolution on RGB image
def convolution_rgb(image, kernel, kernel_center=(-1, -1)):
    red, green, blue = extract_rgb(image)

    red_normalization = normalize(convolve(image=red, kernel=kernel, kernel_center=kernel_center))
    green_normalization = normalize(convolve(image=green, kernel=kernel, kernel_center=kernel_center))
    blue_normalization = normalize(convolve(image=blue, kernel=kernel, kernel_center=kernel_center))

    #cv2.imshow("Extracted Red", red_normalization)
    #cv2.imshow("Extracted green", green_normalization)
    #cv2.imshow("Extracted blue", blue_normalization)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    merged = merge_rgb(red=red_normalization, green=green_normalization, blue=blue_normalization)
    #cv2.imshow("Original image", image)
    #cv2.imshow("Merged image", merged)
    #cv2.waitKey(0)

    #cv2.destroyAllWindows()
    return merged



def find_difference(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    difference = cv2.absdiff(image1, image2)
    difference = normalize(difference)

    return difference


def perform_convolution(imagePath, kernel,kernel_center=(-1, -1)):

    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    out_conv = convolve(image=image, kernel=kernel, kernel_center=kernel_center)
    out_noramlize = normalize(out_conv)

    cv2.imshow('Input image', image)
    cv2.imshow('Covulated image', out_noramlize)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to display the kernel as an image
def display_kernel(kernel):
    plt.imshow(kernel, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Kernel')
    plt.show()

# Input raw images
image_path = '../images/2d_shape.jpg'
image = cv2.imread( image_path )

# Outputs
kernel = None


kernel_type = int(input("Input Kernel Type (1 for Gaussian, 2 for Mean, 3 for Laplacian, 4 for LoG): "))

global c_x, c_y

#Choosing kernel
if kernel_type == 1:
    print("Selected Gaussian Kernel:")
    sig_x = float(input("Enter Sigma(X): "))
    sig_y = float(input("Enter Sigma(Y): "))
    c_x = int(input("Enter Center(X): "))
    c_y = int(input("Enter Center(Y): "))
    kernel = generate_gaussian_kernel(sigmaX=sig_x, sigmaY=sig_y, MUL=3, center_x=c_x, center_y=c_y)
    print(kernel)
    display_kernel(kernel)

elif kernel_type == 2:
    print("Selected Mean Kernel:")
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))
    c_x = int(input("Enter Center(X): "))
    c_y = int(input("Enter Center(Y): "))
    kernel = generate_mean_kernel(rows=rows, cols=cols)
    print(kernel)
    display_kernel(kernel)

elif kernel_type == 3:
    print("Selected Laplacian Kernel:")
    c_x = int(input("Enter Center(X): "))
    c_y = int(input("Enter Center(Y): "))
    negCenter = input("Do you want negative center for Laplacian Kernel? (yes/no): ")
    if negCenter.lower() == "yes":
        kernel = generate_laplacian_kernel(negCenter=True)
    else:
        kernel = generate_laplacian_kernel(negCenter=False)
    print(kernel)
    display_kernel(kernel)

elif kernel_type == 4:
    print("Selected LoG Kernel:")
    c_x = int(input("Enter Center(X): "))
    c_y = int(input("Enter Center(Y): "))
    sig_x = float(input("Enter Sigma(X): "))
    kernel = generate_log_kernel(sigma=sig_x,MUL=3)
    print(kernel)
    display_kernel(kernel)



perform_convolution(imagePath=image_path, kernel=kernel,kernel_center=(c_x, c_y))
img_sobel = perform_sobel(imagePath=image_path, kernel_center=(c_x, c_y))

img1=convolution_rgb(image=image, kernel=kernel, kernel_center=(c_x,c_y))
img2=convolution_hsv(image=image, kernel=kernel, kernel_center=(c_x,c_y))


dif = find_difference(image1=img1, image2=img2)

cv2.imshow("colvolve RGB", img1)
cv2.imshow("convolve HSV", img2)
cv2.imshow("Difference", dif)
cv2.waitKey(0)
