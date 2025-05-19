#Merging two images
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File paths for images and CSV files
file_paths = [
    {
        "image": r"C:\Users\Govind\Downloads\ch2_ohr_ncp_20240425T1406019344_d_img_d18\data\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_d_img_d18.img",
        "csv": r"C:\Users\Govind\Downloads\ch2_ohr_ncp_20240425T1406019344_d_img_d18\geometry\calibrated\20240425\ch2_ohr_ncp_20240425T1406019344_g_grd_d18.csv",
        "height": 93693,
        "width": 12000
    },
    {
        "image": r"C:\Users\Govind\Downloads\ch2_ohr_ncp_20240425T1603031918_d_img_d18\data\calibrated\20240425\ch2_ohr_ncp_20240425T1603031918_d_img_d18.img",
        "csv": r"C:\Users\Govind\Downloads\ch2_ohr_ncp_20240425T1603031918_d_img_d18\geometry\calibrated\20240425\ch2_ohr_ncp_20240425T1603031918_g_grd_d18.csv",
        "height": 90148,
        "width": 12000
    }
]

# Iterate through the list of file paths
for file_info in file_paths:
    image_path = file_info["image"]
    csv_path = file_info["csv"]
    height = file_info["height"]
    width = file_info["width"]

    # Read and reshape the image
    try:
        data = np.fromfile(image_path, dtype=np.uint8).reshape((height, width))
        print(f"Image {image_path} reshaped successfully.")
    except ValueError as e:
        print(f"Error reshaping data for {image_path}: {e}")
        continue

    # Crop a 5000x5000 region from the image
    data_combined = data[0:5000, 0:5000]  # Adjusted crop region

    # Read the CSV file with Scan/Pixel data
    try:
        csv_data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file for {csv_path}: {e}")
        continue

    # Check for necessary columns in the CSV file
    required_columns = ['Scan', 'Pixel', 'Latitude', 'Longitude']
    if not all(col in csv_data.columns for col in required_columns):
        print(f"Error: CSV file {csv_path} must contain {required_columns} columns.")
        continue

    # Extract coordinates within the cropped region (5000x5000)
    coordinate_mappings = []
    for _, row in csv_data.iterrows():
        scan, pixel, lat, lon = row['Scan'], row['Pixel'], row['Latitude'], row['Longitude']
        if 0 <= scan < 5000 and 0 <= pixel < 5000:
            coordinate_mappings.append({'Scan': scan, 'Pixel': pixel, 'Latitude': lat, 'Longitude': lon})

    # Plot the cropped image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(data_combined, cmap='gray')

    # Scatter points for the coordinates
    for coord in coordinate_mappings:
        ax.scatter(coord['Pixel'], coord['Scan'], color='red', s=50, edgecolor='black', linewidth=1)

    # Hover annotation setup
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", 
                        bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(coord):
        """
        Updates the hover annotation.
        """
        annot.xy = (coord['Pixel'], coord['Scan'])
        text = f"Lat: {coord['Latitude']}, Lon: {coord['Longitude']}"
        annot.set_text(text)

    def hover(event):
        """
        Handles hover event to display latitude and longitude.
        """
        if event.inaxes == ax:
            for coord in coordinate_mappings:
                x, y = int(event.xdata), int(event.ydata)
                # Check if hover is near a marker (within 10 pixels)
                if abs(coord['Pixel'] - x) < 10 and abs(coord['Scan'] - y) < 10:
                    update_annot(coord)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            annot.set_visible(False)
            fig.canvas.draw_idle()

    # Connect hover functionality
    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.tight_layout()
    plt.show()