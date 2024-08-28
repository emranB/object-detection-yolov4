import math

def calculate_heading(bbox, image_width, image_height, fov_x, fov_y):
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox
    
    # Step 2: Calculate the center of the bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Step 3: Normalize the coordinates
    x_norm = x_center / image_width
    y_norm = y_center / image_height
    
    # Step 4: Calculate the heading angles
    theta_x = (x_norm - 0.5) * fov_x
    theta_y = (y_norm - 0.5) * fov_y
    
    return theta_x, theta_y

