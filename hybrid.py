#Implementation of lat/long on enhanced img

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the file path for the image and the CSV file
file_path = r"C:\Users\ASUS\OneDrive\Documents\Lunar Illuminate\ch2_ohr_ncp_20240425T1406019344_d_img_d18\data\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_d_img_d18.img"
csv_file_path = r"C:\Users\ASUS\OneDrive\Documents\Lunar Illuminate\ch2_ohr_ncp_20240425T1406019344_d_img_d18\geometry\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_g_grd_d18.csv"

# Assuming you know the correct height and width of the image
height = 93693  # Image height
width = 12000   # Image width

# Read the binary image data using numpy
data = np.fromfile(file_path, dtype=np.uint8)

# Reshape the data into a 2D array
try:
    data = data.reshape((height, width))
    print("Data reshaped successfully.")
except ValueError as e:
    print(f"Error reshaping data: {e}")
    exit()

# Crop a larger 5000x5000 region for analysis
data_combined = data[0:5000, 0:5000]  # Adjusted crop region

# Read the CSV file to get the Scan and Pixel data
try:
    csv_data = pd.read_csv(csv_file_path)  # Read the CSV file
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Ensure the CSV file has the required 'Scan', 'Pixel', 'Latitude', and 'Longitude' columns
required_columns = ['Scan', 'Pixel', 'Latitude', 'Longitude']
if not all(col in csv_data.columns for col in required_columns):
    print(f"Error: CSV file must contain {required_columns} columns.")
    exit()

# Collect coordinates from the 'Scan' and 'Pixel' columns
coordinate_mappings = []

for _, row in csv_data.iterrows():
    scan = row['Scan']  # Row in the image
    pixel = row['Pixel']  # Column in the image
    lat = row['Latitude']
    lon = row['Longitude']

    # Check if scan and pixel are within the 5000x5000 cropped image bounds
    if 0 <= scan < 5000 and 0 <= pixel < 5000:
        coordinate_mappings.append({
            'Scan': scan,
            'Pixel': pixel,
            'Latitude': lat,
            'Longitude': lon
        })

# Convert to DataFrame for easy inspection
coordinate_df = pd.DataFrame(coordinate_mappings)

# Display the combined lunar surface image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(data_combined, cmap='gray')

# Plot the Scan/Pixel markers on the combined lunar surface image
for coord in coordinate_mappings:
    scan = coord['Scan']
    pixel = coord['Pixel']

    # Ensure the coordinates are within the bounds of the combined image
    ax.scatter(pixel, scan, color='red', s=50, edgecolor='black', linewidth=1)

# Add hover event to show latitude and longitude
annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                    textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind, lat, lon):
    """
    Update annotation text and position based on hover.
    """
    pos = ind["ind"][0]
    annot.xy = (coordinate_mappings[pos]['Pixel'], coordinate_mappings[pos]['Scan'])
    text = f"Lat: {lat}, Lon: {lon}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    """
    Function to handle mouse hover and display lat/lon.
    """
    vis = annot.get_visible()
    if event.inaxes == ax:
        for coord in coordinate_mappings:
            x, y = int(event.xdata), int(event.ydata)
            
            # Check if the event (mouse hover) is near a marker (within a few pixels)
            if abs(coord['Pixel'] - x) < 10 and abs(coord['Scan'] - y) < 10:
                update_annot({'ind': [coordinate_mappings.index(coord)]}, coord['Latitude'], coord['Longitude'])
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

# Connect the hover function to the matplotlib event manager
fig.canvas.mpl_connect("motion_notify_event", hover)

plt.tight_layout()
plt.show()