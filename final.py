#Longitude and latitude included user image

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from geopy.distance import great_circle

# Define file paths
file_path = r"C:\Users\ASUS\OneDrive\Documents\Lunar Illuminate\ch2_ohr_ncp_20240425T1406019344_d_img_d18\data\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_d_img_d18.img"
csv_file_path = r"C:\Users\ASUS\OneDrive\Documents\Lunar Illuminate\ch2_ohr_ncp_20240425T1406019344_d_img_d18\geometry\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_g_grd_d18.csv"

# Define image dimensions and cropping parameters
height = 93693
width = 12000
x_crop_start = 3000
y_crop_start = 3000
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

# Read the CSV file to get the Scan and Pixel data
try:
    csv_data = pd.read_csv(csv_file_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Ensure the CSV file has the required columns
required_columns = ['Scan', 'Pixel', 'Latitude', 'Longitude']
if not all(col in csv_data.columns for col in required_columns):
    print(f"Error: CSV file must contain {required_columns} columns.")
    exit()

# Collect coordinates from the CSV file
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

# Convert to DataFrame
coordinate_df = pd.DataFrame(coordinate_mappings)

# Distance calculation function using the Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    return great_circle((lat1, lon1), (lat2, lon2)).kilometers

# Update function with sliders for dynamic adjustment
def update(val):
    corrected_image = correct_illumination(cropped_image)
    clahe_image = apply_clahe(corrected_image)
    enhanced_image = enhance_craters(clahe_image, gamma=gamma_slider.val, exposure=exposure_slider.val, contrast=contrast_slider.val)
    
    ax[0].imshow(cropped_image, cmap='gray')
    ax[0].set_title('Raw Cropped Image')

    ax[1].clear()
    ax[1].imshow(enhanced_image)
    for coord in coordinate_mappings:
        ax[1].scatter(coord['Pixel'], coord['Scan'], color='red', s=50, edgecolor='black', linewidth=1, alpha=0.01)

    
    fig.canvas.draw_idle()

# Create the plot and sliders
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.9)

# Display the raw image initially
ax[0].imshow(cropped_image, cmap='gray')
ax[0].set_title('Raw Cropped Image')

# Define sliders for enhancement parameters with initial values
ax_slider_gamma = plt.axes([0.1, 0.28, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_exposure = plt.axes([0.1, 0.22, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_contrast = plt.axes([0.1, 0.16, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Initialize the sliders with specific values
gamma_slider = Slider(ax_slider_gamma, 'Gamma', 0.1, 2.0, valinit=0.6, valstep=0.1)
exposure_slider = Slider(ax_slider_exposure, 'Exposure', 0.5, 2.0, valinit=1.0, valstep=0.1)
contrast_slider = Slider(ax_slider_contrast, 'Contrast', 0.5, 2.0, valinit=1.5, valstep=0.1)

# Initialize the plot with the default values
update(None)

# Hover event function
def update_annot(ind, lat, lon):
    pos = ind["ind"][0]
    annot.xy = (coordinate_mappings[pos]['Pixel'], coordinate_mappings[pos]['Scan'])
    text = f"Lat: {lat}, Lon: {lon}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax[1]:
        for coord in coordinate_mappings:
            x, y = int(event.xdata), int(event.ydata)
            if abs(coord['Pixel'] - x) < 10 and abs(coord['Scan'] - y) < 10:
                update_annot({'ind': [coordinate_mappings.index(coord)]}, coord['Latitude'], coord['Longitude'])
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

# Create annotation for hover display
annot = ax[1].annotate("", xy=(0, 0), xytext=(20, 20),
                       textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                       arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# Add functionality to select two points and calculate the distance
selected_coords = []

def on_click(event):
    if event.inaxes == ax[1]:
        x, y = int(event.xdata), int(event.ydata)
        for coord in coordinate_mappings:
            if abs(coord['Pixel'] - x) < 10 and abs(coord['Scan'] - y) < 10:
                selected_coords.append((coord['Latitude'], coord['Longitude']))
                if len(selected_coords) == 2:
                    lat1, lon1 = selected_coords[0]
                    lat2, lon2 = selected_coords[1]
                    distance = calculate_distance(lat1, lon1, lat2, lon2)
                    print(f"Distance between points: {distance:.2f} kilometers")
                    selected_coords.clear()  # Clear the selected points

# Connect the hover function and click event to the matplotlib event manager
fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.mpl_connect("button_press_event", on_click)

# Attach update function to sliders
gamma_slider.on_changed(update)
exposure_slider.on_changed(update)
contrast_slider.on_changed(update)

plt.show()