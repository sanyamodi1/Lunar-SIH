import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from geopy.distance import great_circle

# Define file paths for two datasets
file_path1 = r"C:\Users\Govind\Downloads\ch2_ohr_ncp_20240425T1603031918_d_img_d18\data\calibrated\20240425\ch2_ohr_ncp_20240425T1603031918_d_img_d18.img"
csv_file_path1 = r"C:\Users\Govind\Downloads\ch2_ohr_ncp_20240425T1603031918_d_img_d18\geometry\calibrated\20240425\ch2_ohr_ncp_20240425T1603031918_g_grd_d18.csv"

file_path2 = r"C:\Users\Govind\Downloads\ch2_ohr_ncp_20240425T1406019344_d_img_d18\data\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_d_img_d18.img"
csv_file_path2 = r"C:\Users\Govind\Downloads\ch2_ohr_ncp_20240425T1406019344_d_img_d18\geometry\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_g_grd_d18.csv"

# Define image dimensions and cropping parameters
height1, width1 = 90148, 12000
x_crop_start1, y_crop_start1, crop_width1, crop_height1 = 3000, 3000, 1000, 1000

height2, width2 = 93693, 12000
x_crop_start2, y_crop_start2, crop_width2, crop_height2 = 3000, 3000, 1000, 1000

# Load and crop the images
def load_and_crop_image(file_path, height, width, x_crop_start, y_crop_start, crop_width, crop_height):
    data = np.fromfile(file_path, dtype=np.uint8)
    data = data.reshape((height, width))
    return data[y_crop_start:y_crop_start+crop_height, x_crop_start:x_crop_start+crop_width]

cropped_image1 = load_and_crop_image(file_path1, height1, width1, x_crop_start1, y_crop_start1, crop_width1, crop_height1)
cropped_image2 = load_and_crop_image(file_path2, height2, width2, x_crop_start2, y_crop_start2, crop_width2, crop_height2)

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

# Enhance the craters in the image
def enhance_craters(image, gamma=0.6, exposure=1.0, contrast=1.5):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    enhanced_image = adjust_gamma(output_image, gamma=gamma)
    enhanced_image = np.clip(enhanced_image * exposure, 0, 255)
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=contrast, beta=0)
    sharpened_image = sharpen_image(enhanced_image)
    return sharpened_image

# Sharpen the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Read CSV files and collect coordinates
def load_coordinates(csv_file_path, crop_width, crop_height):
    try:
        csv_data = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    required_columns = ['Scan', 'Pixel', 'Latitude', 'Longitude']
    if not all(col in csv_data.columns for col in required_columns):
        print(f"Error: CSV file must contain {required_columns} columns.")
        return []

    coordinate_mappings = []
    for _, row in csv_data.iterrows():
        scan = row['Scan']
        pixel = row['Pixel']
        lat = row['Latitude']
        lon = row['Longitude']
        if 0 <= scan < crop_height and 0 <= pixel < crop_width:
            coordinate_mappings.append({
                'Scan': scan,
                'Pixel': pixel,
                'Latitude': lat,
                'Longitude': lon
            })
    return coordinate_mappings

coordinate_mappings1 = load_coordinates(csv_file_path1, crop_width1, crop_height1)
coordinate_mappings2 = load_coordinates(csv_file_path2, crop_width2, crop_height2)

# Convert coordinates to NumPy array
coords2 = np.array([(coord['Pixel'], coord['Scan']) for coord in coordinate_mappings2])
lat_lon2 = np.array([(coord['Latitude'], coord['Longitude']) for coord in coordinate_mappings2])

# Distance calculation function using the Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    return great_circle((lat1, lon1), (lat2, lon2)).kilometers

# Find nearest neighbor using NumPy
def find_nearest_neighbor(x, y):
    distances = np.sqrt((coords2[:, 0] - x)*2 + (coords2[:, 1] - y)*2)
    nearest_index = np.argmin(distances)
    return nearest_index

# Calculate SNR
def calculate_snr(image):
    signal_mean = np.mean(image)
    noise = image - cv2.GaussianBlur(image, (25, 25), 0)
    noise_std = np.std(noise)
    return signal_mean / noise_std if noise_std != 0 else 0

# Update function with sliders for dynamic adjustment and SNR calculation
def update(val):
    corrected_image = correct_illumination(cropped_image2)
    clahe_image = apply_clahe(corrected_image)
    enhanced_image = enhance_craters(clahe_image, gamma=gamma_slider.val, exposure=exposure_slider.val, contrast=contrast_slider.val)
    
    # Calculate SNR for both the raw and enhanced images
    snr_raw = calculate_snr(cropped_image2)
    snr_enhanced = calculate_snr(enhanced_image)
    
    ax[0].imshow(cropped_image2, cmap='gray')
    ax[0].set_title(f'Raw Cropped Image (SNR: {snr_raw:.2f})')

    ax[1].clear()
    ax[1].imshow(enhanced_image)
    ax[1].set_title(f'Enhanced Image (SNR: {snr_enhanced:.2f})')
    
    # Plot coordinates on enhanced image
    for coord in coordinate_mappings2:
        ax[1].scatter(coord['Pixel'], coord['Scan'], color='red', s=50, edgecolor='black', linewidth=1, alpha=0.01)

    fig.canvas.draw_idle()

# Create the plot and sliders
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.9)

# Display the raw image initially
ax[0].imshow(cropped_image2, cmap='gray')
ax[0].set_title('Raw Cropped Image')

# Define sliders for enhancement parameters with initial values
ax_slider_gamma = plt.axes([0.1, 0.28, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_exposure = plt.axes([0.1, 0.22, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_contrast = plt.axes([0.1, 0.16, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Initialize the sliders with specific values
gamma_slider = Slider(ax_slider_gamma, 'Gamma', 0.1, 3.0, valinit=0.6, valstep=0.1)
exposure_slider = Slider(ax_slider_exposure, 'Exposure', 0.1, 3.0, valinit=1.0, valstep=0.1)
contrast_slider = Slider(ax_slider_contrast, 'Contrast', 0.1, 3.0, valinit=1.5, valstep=0.1)

# Attach the update function to sliders
gamma_slider.on_changed(update)
exposure_slider.on_changed(update)
contrast_slider.on_changed(update)

# Initial update to display images with default slider values
update(None)

# Add text annotation for latitude coordinates
text_annotation = ax[1].text(0.5, 0.95, '', transform=ax[1].transAxes, fontsize=12, ha='center', va='top', color='white', bbox=dict(facecolor='black', alpha=0.7))

def on_move(event):
    if event.inaxes == ax[1]:
        x, y = int(event.xdata), int(event.ydata)
        nearest_index = find_nearest_neighbor(x, y)
        if nearest_index is not None:
            lat, lon = lat_lon2[nearest_index]
            text_annotation.set_text(f'Latitude: {lat:.6f}, Longitude: {lon:.6f}')
            text_annotation.set_visible(True)
            fig.canvas.draw_idle()
        else:
            text_annotation.set_visible(False)
            fig.canvas.draw_idle()

# Connect the on_move function to the mouse motion event
fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()