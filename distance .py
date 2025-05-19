#Calculating distance 

import numpy as np

# Moon's radius in kilometers
MOON_RADIUS = 1737.4

# Haversine formula to calculate distance between two points on the Moon
def calculate_lunar_distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in kilometers
    distance = MOON_RADIUS * c
    return distance

# Example usage
lat1, lon1 = 10.0, 45.0  # Coordinates of point 1 (in degrees)
lat2, lon2 = 15.0, 50.0  # Coordinates of point 2 (in degrees)

distance = calculate_lunar_distance(lat1, lon1, lat2, lon2)
print(f"Distance between points: {distance:.2f} kilometers")
                                                                                                                                                                                                                                                                                                                                 