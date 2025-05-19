import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load the image data
file_path = r"C:\Users\ASUS\OneDrive\Documents\Lunar Illuminate\ch2_ohr_ncp_20240425T1406019344_d_img_d18\data\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_d_img_d18.img"

# Define image dimensions
height = 93693
width = 12000

# Define cropping parameters
x_crop_start = 5000
y_crop_start = 45000
crop_width = 1000
crop_height = 1000

# Load and crop the image
data = np.fromfile(file_path, dtype=np.uint8)
data = data.reshape((height, width))
cropped_image = data[y_crop_start:y_crop_start+crop_height, x_crop_start:x_crop_start+crop_width]

# Apply CLAHE for shadow reduction
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

# Apply illumination correction
def correct_illumination(image, blur_kernel_size=25):
    background = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    corrected_image = cv2.addWeighted(image, 1.5, background, -0.5, 0)
    return corrected_image

# Gamma correction function
def adjust_gamma(image, gamma=1.0):
    image = np.clip(image, 0, 255).astype(np.uint8)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Enhance only the craters detected
def enhance_craters(image, gamma=0.5, exposure=1.2, contrast=1.3):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    
    # Apply gamma correction
    enhanced_image = adjust_gamma(output_image, gamma=gamma)
    
    # Apply exposure enhancement
    enhanced_image = np.clip(enhanced_image * exposure, 0, 255)
    
    # Apply contrast
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=contrast, beta=0)
    
    # Sharpen the final output image
    sharpened_image = sharpen_image(enhanced_image)
    
    return sharpened_image

# Sharpen image function
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Update function with sliders for dynamic adjustment
def update(val):
    # Preprocessing: Apply CLAHE and illumination correction
    corrected_image = correct_illumination(cropped_image)
    clahe_image = apply_clahe(corrected_image)
    
    # Apply crater enhancement
    enhanced_image = enhance_craters(clahe_image, gamma=gamma_slider.val, exposure=exposure_slider.val, contrast=contrast_slider.val)
    
    # Update the displayed raw and enhanced images
    ax[0].imshow(cropped_image, cmap='gray')
    ax[0].set_title('Raw Cropped Image')
    
    ax[1].imshow(enhanced_image)
    ax[1].set_title('Enhanced Image')
    
    fig.canvas.draw_idle()

# Create the plot and sliders
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.9)

# Display the raw image initially
ax[0].imshow(cropped_image, cmap='gray')
ax[0].set_title('Raw Cropped Image')

# Define sliders for enhancement parameters
ax_slider_gamma = plt.axes([0.1, 0.28, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_exposure = plt.axes([0.1, 0.22, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_contrast = plt.axes([0.1, 0.16, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Initialize the sliders
gamma_slider = Slider(ax_slider_gamma, 'Gamma', 0.1, 2.0, valinit=0.5, valstep=0.1)
exposure_slider = Slider(ax_slider_exposure, 'Exposure', 0.5, 2.0, valinit=1.2, valstep=0.1)
contrast_slider = Slider(ax_slider_contrast, 'Contrast', 0.5, 2.0, valinit=1.3, valstep=0.1)

# Initialize the plot with the default values
update(None)

# Attach update function to sliders
gamma_slider.on_changed(update)
exposure_slider.on_changed(update)
contrast_slider.on_changed(update)

plt.show()