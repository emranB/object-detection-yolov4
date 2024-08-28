import numpy as np
from sort.sort import Sort

# Create an instance of the tracker
tracker = Sort()

# Example detections: [x_min, y_min, x_max, y_max, confidence]
detections = np.array([
    [100, 200, 150, 250, 0.9],
    [300, 400, 350, 450, 0.8]
])

# Update the tracker with new detections
tracked_objects = tracker.update(detections)

print(tracked_objects)
