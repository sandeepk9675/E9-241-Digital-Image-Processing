import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy import misc
from matplotlib.image import imread
from math import pi



def gaussian_kernel(size, sigma):
    """Creates a 2D Gaussian kernel."""
    kernel = np.zeros((size, size))
    center = size // 2

    if sigma == 0:
        kernel[center, center] = 1
        return kernel
        
    for x in range(size):
        for y in range(size):
            x_dist = (x - center) ** 2
            y_dist = (y - center) ** 2
            kernel[x, y] = np.exp(-(x_dist + y_dist) / (2 * sigma ** 2))
    
    # Normalize the kernel so that the sum equals 1
    kernel /= np.sum(kernel)
    
    return kernel


def Histogram(img):

    if np.max(img) <= 1:
        img = (255 * img).astype(int)
    # Count occurrences for each intensity level
    img_freq = np.zeros(256, dtype=int)
    for l in range(256):
       img_freq[l] = np.count_nonzero(img == l)

    return img_freq


def Within_class_variance(img):

    # Normalize the image to 0-255 range
    if np.max(img) <= 1:
        img = (255 * img).astype(int)
    
    var_w_lst = np.zeros(256, dtype = int)
    img_freq = Histogram(img)
    #print(coins_array)
    w_t = np.sum(img_freq)


    for t in range(256):
        mean_0 = 0
        mean_1 = 0
        var_0 = 0
        var_1 = 0
        w_0 = 0
        w_1 = 0
        for i in range(t + 1):
            w_0 += img_freq[i]

        for j in range(t + 1, 256):
            w_1 += img_freq[j]

        if  w_0 > 0:
            for i in range(t + 1):
                mean_0 += i * img_freq[i] / w_0
        if t < 255 and w_1 > 0:
            for j in range(t + 1, 256):
                mean_1 += j * img_freq[j] / w_1


        if  w_0 > 0:
            for i in range(t + 1):
                var_0 += ((i-mean_0)**2) * img_freq[i] / w_0
        if t < 255 and w_1 > 0:
            for j in range(t + 1, 256):
                var_1 += ((j-mean_1)**2) * img_freq[j] / w_1
        var_w = int((w_0*var_0 + w_1*var_1)/w_t)
        var_w_lst[t] = var_w
        
    return var_w_lst


# Load the original image (RGB)
img = mpimg.imread("moon_noisy.png")
print("Image Shape:", img.shape)

# Split image into red, green, and blue channels
red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]

# Create subplots
fig, axs1 = plt.subplots(2, 4, figsize=(20, 10))
fig, axs2 = plt.subplots(2, 4, figsize=(20, 10))
fig, axs3 = plt.subplots(2, 4, figsize=(20, 10))

# Apply Gaussian filters with different variances
n = 0
for variance in [0, 0.1, 0.5, 1, 2.5, 5, 10, 20]:
    kernel_size = 41
    sigma_g = variance
    gaussian_filter = gaussian_kernel(kernel_size, sigma_g)

    # Convolve the filter with each channel
    red_blurred_channel = convolve(red_channel, gaussian_filter)
    green_blurred_channel = convolve(green_channel, gaussian_filter)
    blue_blurred_channel = convolve(blue_channel, gaussian_filter)

    # Stack the channels back into an RGB image
    blurred_img = np.dstack((red_blurred_channel, green_blurred_channel, blue_blurred_channel))
    # Convert the blurred image to grayscale
    blurred_img_gray = np.dot(blurred_img, [0.2989, 0.5870, 0.1140])

    #binarize the blurred image
    within_variance = Within_class_variance(blurred_img_gray)
    optimal_within_variance = np.min(within_variance)
    threshold = np.argmin(within_variance)
    print("Threshold: {} \noptimal within-class variances for σ({})=:{}".format(threshold, variance, optimal_within_variance))
    binarized_img = np.zeros_like(blurred_img_gray)
    for i in range(binarized_img.shape[0]):
        for j in range(binarized_img.shape[1]):
            if blurred_img_gray[i, j] > threshold/255:
                binarized_img[i, j] = 255
            else:
                binarized_img[i, j] = 0

    # Plot the blurred image in axs1
    row = n // 4
    col = n % 4

    axs1[row, col].imshow(blurred_img)
    axs1[row, col].set_title(f"Blurred (σ_g={variance})")
    axs1[row, col].axis('off')

    # Calculate the histogram for the grayscale blurred image
    hist_values = Histogram(blurred_img_gray)

    # Plot the histogram in axs2
    axs2[row, col].plot(np.arange(0, 256), hist_values)
    axs2[row, col].set_title(f"Histogram (σ_g={variance})")
    axs2[row, col].set_xlim([0, 255])
    
    axs3[row, col].imshow(binarized_img, cmap='gray')
    axs3[row, col].set_title(f"Binarized image (σ_g={variance})")
    axs3[row, col].axis('off')

    n += 1

# Show the plots
plt.show()




def downsampling(img, factor):
    size = img.shape

    sampled_img = np.zeros((size[0]//factor, size[1]//factor), dtype=img.dtype)
    for i in range(size[0]//factor):
        for j in range(size[1]//factor):
            sampled_img[i, j] = img[i*factor, j*factor]
    return sampled_img




# write a function to upsample the image by a factor of 3

def upsampling(img, factor):
    size1 = img.shape  # original image size (113, 200)
    upsampled_img = np.zeros((int(size1[0] * factor), int(size1[1] * factor)), dtype=img.dtype)
    size = upsampled_img.shape  # upsampled image size
    
    for i in range(size[0]):  # Iterate over the rows of upsampled image
        for j in range(size[1]):  # Iterate over the columns of upsampled image
            a_i = i / factor  # Corresponding row in the original image
            a_j = j / factor  # Corresponding column in the original image
            
            if a_i == int(a_i) and a_j == int(a_j):
                # Direct copy from the original image
                upsampled_img[i, j] = img[int(a_i), int(a_j)]
            elif a_i != int(a_i) and a_j != int(a_j):
                # Bilinear interpolation
                if int(a_i) + 1 < size1[0] and int(a_j) + 1 < size1[1]:
                    interpolation_x1 = img[int(a_i), int(a_j)+1] + (img[int(a_i)+1, int(a_j)+1] - img[int(a_i), int(a_j)+1]) * (a_i - int(a_i))
                    interpolation_x2 = img[int(a_i), int(a_j)] + (img[int(a_i)+1, int(a_j)] - img[int(a_i), int(a_j)]) * (a_i - int(a_i))
                    interpolation_y1 = interpolation_x2 + (interpolation_x1 - interpolation_x2) * (a_j - int(a_j))
                    upsampled_img[i, j] = interpolation_y1
            elif a_i == int(a_i) and a_j != int(a_j):
                # Horizontal interpolation
                if int(a_j) + 1 < size1[1]:
                    interpolation_y1 = img[int(a_i), int(a_j)] + (img[int(a_i), int(a_j)+1] - img[int(a_i), int(a_j)]) * (a_j - int(a_j))
                    upsampled_img[i, j] = interpolation_y1
            elif a_i != int(a_i) and a_j == int(a_j):
                # Vertical interpolation
                if int(a_i) + 1 < size1[0]:
                    interpolation_x1 = img[int(a_i), int(a_j)] + (img[int(a_i)+1, int(a_j)] - img[int(a_i), int(a_j)]) * (a_i - int(a_i))
                    upsampled_img[i, j] = interpolation_x1

    return upsampled_img


# Load and display the original image
img_1 = mpimg.imread("flowers.png")

downsampled_img = downsampling(img_1, 2)

# Create subplots with 1 row and 2 columns
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
ax[0].imshow(img_1, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')  # Hide the axis
print("Original image shape:", img_1.shape)

# Display the downsampled image
downsampled_img = downsampling(img_1, 2)
ax[1].imshow(downsampled_img, cmap='gray')
ax[1].set_title('Downsampled Image')
ax[1].axis('off')  # Hide the axis
print("Downsampled image shape:", downsampled_img.shape)

# Display the images
plt.show()


# Question 2(a)
down_up_sampled_img = upsampling(downsampled_img, 3)
print("Downsampled followed by upsampled image shape:", down_up_sampled_img.shape)

# Save the downsampled followed by upsampled image
mpimg.imsave("down_up_sampled.png", down_up_sampled_img, cmap='gray')

# Question 2(b)
up_sampled_img = upsampling(img_1, 1.5)
print("Upsampled image shape:", up_sampled_img.shape)

# Save the upsampled image
mpimg.imsave("up_sampled.png", up_sampled_img, cmap='gray')


# Create subplots to plot both the upsampled images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the downsampled followed by upsampled image
ax[0].imshow(down_up_sampled_img, cmap='gray')
ax[0].set_title('Downsampled & Upsampled Image')
ax[0].axis('off')  # Hide the axis

# Display the upsampled image
ax[1].imshow(up_sampled_img, cmap='gray')
ax[1].set_title('Upsampled Image')
ax[1].axis('off')  # Hide the axis

# Show the plot
plt.show()


# Function to display images using matplotlib
def display_images(original, adjusted, title1="Original", title2="Adjusted"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(adjusted)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.show()


# Adjust brightness
def brightnessAdjust(fname, p):
    # Read the image
    img = imread(fname)
    
    # p: brightness factor (0 <= p <= 1)
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    
    # Adjust brightness
    else:
        return np.clip(2*(p-0.5)*255 +img, 0, 255).astype(np.uint8)



def contrastAdjust(img, p):
    # Ensure p is between 0 and 1
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    
    # Calculate the mean for each channel (R, G, B) separately
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    theta = pi/4 + pi/2*(p-0.5)
    p_slope = np.tan(theta)
    y = p_slope * (img - mean) + mean
    adjusted_img = np.clip(y, 0, 255).astype(np.uint8)

    return adjusted_img


# Load an image file
original_image = imread('brightness_contrast.jpg')

# Create two subplots for brightness and contrast adjustments
fig1, axs1 = plt.subplots(2, 5, figsize=(20, 10))  # For brightness adjustments
fig2, axs2 = plt.subplots(2, 5, figsize=(20, 10))  # For contrast adjustments

n = 0

for p in [0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    # Adjust brightness and contrast
    brightness_img = brightnessAdjust('brightness_contrast.jpg', p)
    contrast_img = contrastAdjust(original_image, p)  # Pass the original image array here

    row = n // 5  # Adjust row and col calculation based on 5 columns
    col = n % 5
    
    # Display brightness-adjusted images
    axs1[row, col].imshow(brightness_img)
    axs1[row, col].set_title(f"Brightness (p={p})")
    axs1[row, col].axis('off')
    
    # Display contrast-adjusted images
    axs2[row, col].imshow(contrast_img)
    axs2[row, col].set_title(f"Contrast (p={p})")
    axs2[row, col].axis('off')
    
    n += 1  # Increment counter

# Show the plots
plt.show()
