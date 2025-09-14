# skin_lesion_image_filters.py
# Python implementation of smoothing and sharpening filters for skin lesion images
# Includes Gaussian and Salt & Pepper noise, and grayscale/RGB sharpening

from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

# ------------------------------
# STEP 1: Upload Image
# ------------------------------
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Read original color image
img_color = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Selected Image")
plt.axis("off")
plt.show()

# ------------------------------
# STEP 2: Preprocessing
# ------------------------------
# Convert to grayscale and resize
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
gray_resized = cv2.resize(gray, (256, 256))
color_resized = cv2.resize(img_rgb, (256, 256))
gray_normalized = gray_resized / 255.0

# Display preprocessing results
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(gray_resized, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(gray_normalized, cmap='gray')
plt.title("Normalized Image")
plt.axis("off")
plt.show()

# ------------------------------
# STEP 3: Noise Functions
# ------------------------------
def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, prob=0.02):
    noisy = np.copy(image)
    # Salt
    num_salt = int(np.ceil(prob * image.size * 0.5))
    coords = (np.random.randint(0, image.shape[0], num_salt),
              np.random.randint(0, image.shape[1], num_salt))
    noisy[coords] = 255
    # Pepper
    num_pepper = int(np.ceil(prob * image.size * 0.5))
    coords = (np.random.randint(0, image.shape[0], num_pepper),
              np.random.randint(0, image.shape[1], num_pepper))
    noisy[coords] = 0
    return noisy

# ------------------------------
# STEP 4: Smoothing Filters
# ------------------------------
def mean_filter(image, ksize=3):
    return cv2.blur(image, (ksize, ksize))

def median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def mode_filter(image, ksize=3):
    pad = ksize // 2
    padded = np.pad(image, pad, mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+ksize, j:j+ksize].flatten()
            output[i, j] = mode(region, keepdims=False)[0]
    return output

# ------------------------------
# STEP 5: Apply Noise + Filters
# ------------------------------
mean_orig = mean_filter(gray_resized)
median_orig = median_filter(gray_resized)
mode_orig = mode_filter(gray_resized)

gauss_noisy = add_gaussian_noise(gray_resized)
mean_gauss = mean_filter(gauss_noisy)
median_gauss = median_filter(gauss_noisy)
mode_gauss = mode_filter(gauss_noisy)

sp_noisy = add_salt_pepper_noise(gray_resized)
mean_sp = mean_filter(sp_noisy)
median_sp = median_filter(sp_noisy)
mode_sp = mode_filter(sp_noisy)

# ------------------------------
# STEP 6: Plot Smoothing Filter Results
# ------------------------------
plt.figure(figsize=(15, 10))
plt.suptitle("Comparison of Smoothing Filters", fontsize=20, fontweight='bold', y=0.95)

# Row 1: Original
plt.subplot(3,5,1); plt.imshow(gray_resized, cmap='gray'); plt.title("Original"); plt.axis("off")
plt.subplot(3,5,2); plt.imshow(mean_orig, cmap='gray'); plt.title("Mean Filter"); plt.axis("off")
plt.subplot(3,5,3); plt.imshow(median_orig, cmap='gray'); plt.title("Median Filter"); plt.axis("off")
plt.subplot(3,5,4); plt.imshow(mode_orig, cmap='gray'); plt.title("Mode Filter"); plt.axis("off")
plt.subplot(3,5,5); plt.text(0.5,0.5,"Original + Filters", fontsize=14, ha='center'); plt.axis("off")

# Row 2: Gaussian
plt.subplot(3,5,6); plt.imshow(gauss_noisy, cmap='gray'); plt.title("Gaussian Noise"); plt.axis("off")
plt.subplot(3,5,7); plt.imshow(mean_gauss, cmap='gray'); plt.title("Mean Filter"); plt.axis("off")
plt.subplot(3,5,8); plt.imshow(median_gauss, cmap='gray'); plt.title("Median Filter"); plt.axis("off")
plt.subplot(3,5,9); plt.imshow(mode_gauss, cmap='gray'); plt.title("Mode Filter"); plt.axis("off")
plt.subplot(3,5,10); plt.text(0.5,0.5,"Gaussian Noise + Filters", fontsize=14, ha='center'); plt.axis("off")

# Row 3: Salt & Pepper
plt.subplot(3,5,11); plt.imshow(sp_noisy, cmap='gray'); plt.title("Salt & Pepper Noise"); plt.axis("off")
plt.subplot(3,5,12); plt.imshow(mean_sp, cmap='gray'); plt.title("Mean Filter"); plt.axis("off")
plt.subplot(3,5,13); plt.imshow(median_sp, cmap='gray'); plt.title("Median Filter"); plt.axis("off")
plt.subplot(3,5,14); plt.imshow(mode_sp, cmap='gray'); plt.title("Mode Filter"); plt.axis("off")
plt.subplot(3,5,15); plt.text(0.5,0.5,"Salt & Pepper Noise + Filters", fontsize=14, ha='center'); plt.axis("off")

plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0.4)
plt.show()

# ------------------------------
# STEP 7: Sharpening Filters
# ------------------------------
# Grayscale Sobel and Laplacian
sobel_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1,0, ksize=3)
sobel_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0,1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_display = cv2.convertScaleAbs(sobel_combined)

laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
laplacian_display = cv2.convertScaleAbs(laplacian)

combined_gray = cv2.addWeighted(sobel_display, 0.5, laplacian_display, 0.5, 0)

# RGB channel sharpening
r,g,b = cv2.split(color_resized)

def apply_sobel_laplacian(channel):
    sobel_x = cv2.Sobel(channel, cv2.CV_64F,1,0,ksize=3)
    sobel_y = cv2.Sobel(channel, cv2.CV_64F,0,1,ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel)
    laplacian = cv2.Laplacian(channel, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    combined = cv2.addWeighted(sobel,0.5,laplacian,0.5,0)
    return sobel, laplacian, combined

sobel_r, lap_r, comb_r = apply_sobel_laplacian(r)
sobel_g, lap_g, comb_g = apply_sobel_laplacian(g)
sobel_b, lap_b, comb_b = apply_sobel_laplacian(b)

sobel_rgb = cv2.merge([sobel_r, sobel_g, sobel_b])
lap_rgb = cv2.merge([lap_r, lap_g, lap_b])
combined_rgb = cv2.merge([comb_r, comb_g, comb_b])

# Display sharpening results
plt.figure(figsize=(16,10))

plt.subplot(2,4,1); plt.imshow(gray_resized, cmap='gray'); plt.title("Gray Original"); plt.axis("off")
plt.subplot(2,4,2); plt.imshow(sobel_display, cmap='gray'); plt.title("Sobel Gray"); plt.axis("off")
plt.subplot(2,4,3); plt.imshow(laplacian_display, cmap='gray'); plt.title("Laplacian Gray"); plt.axis("off")
plt.subplot(2,4,4); plt.imshow(combined_gray, cmap='gray'); plt.title("Sobel+Laplacian Gray"); plt.axis("off")

plt.subplot(2,4,5); plt.imshow(color_resized); plt.title("Original RGB"); plt.axis("off")
plt.subplot(2,4,6); plt.imshow(sobel_rgb); plt.title("Sobel RGB"); plt.axis("off")
plt.subplot(2,4,7); plt.imshow(lap_rgb); plt.title("Laplacian RGB"); plt.axis("off")
plt.subplot(2,4,8); plt.imshow(combined_rgb); plt.title("Sobel+Laplacian RGB"); plt.axis("off")

plt.tight_layout()
plt.show()
