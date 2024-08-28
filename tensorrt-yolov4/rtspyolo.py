"""rtspyolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import os
import time
import argparse
from datetime import datetime

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.rtspcam import Camera  # Import the Camera class for RTSP
from utils.display import open_window, set_display, show_fps, FpsCalculator
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


from heading import calculate_heading
import numpy as np

from sort import Sort 

WINDOW_NAME = 'TrtYOLODemo'
BASE_OUTPUT_DIR = 'detections/crop'  # Base directory for output
OUTPUT_DIR = ''  # This will be set to a new subdirectory for each run

# Set detection accuracy threshold directly in the code
DETECTION_ACCURACY_THRESHOLD = 0.5

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    # parser = add_camera_args(parser)  # Ensure this method is properly imported or defined
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '--rtsp_url', type=str, required=True,
        help='RTSP URL of the camera')
    args = parser.parse_args()
    return args

def create_new_experiment_dir(base_dir):
    """Create a new subdirectory under the base directory with an incremental number."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    exp_numbers = [int(d.replace('exp', '')) for d in existing_dirs if d.startswith('exp') and d.replace('exp', '').isdigit()]
    next_exp_num = max(exp_numbers, default=0) + 1
    new_exp_dir = os.path.join(base_dir, f'exp{next_exp_num}')
    os.makedirs(new_exp_dir)
    return new_exp_dir

def save_cropped_images(img, boxes, clss, confs, cls_dict):
    """Save cropped images of detections.
    
    # Arguments
      img: the original image.
      boxes: bounding box coordinates of detected objects.
      clss: class IDs of detected objects.
      confs: confidence scores of detected objects.
      cls_dict: dictionary mapping class IDs to class names.
    """
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < DETECTION_ACCURACY_THRESHOLD:
            continue
        cls_id = clss[i]
        cls_name = cls_dict.get(cls_id, 'CLS{}'.format(cls_id))
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_img = img[y_min:y_max, x_min:x_max]
        class_dir = os.path.join(OUTPUT_DIR, cls_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(class_dir, f'{cls_name}_{timestamp}_{conf:.2f}.jpg')
        cv2.imwrite(filename, cropped_img)

def loop_and_detect(cam, trt_yolo, conf_th, vis, allowed_classes, cls_dict):
    """Continuously capture images from the camera and perform object detection with SORT tracking.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      allowed_classes: list of class names to detect.
      cls_dict: dictionary mapping class IDs to class names.
    """
    if allowed_classes is None:
        allowed_classes = list(cls_dict.values())
    allowed_class_ids = [cls_id for cls_id, cls_name in cls_dict.items() if cls_name in allowed_classes]

    full_scrn = False
    fps_calculator = FpsCalculator()  # For FPS calculation

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec can be changed as needed
    video_output_path = os.path.join(OUTPUT_DIR, 'output_video.avi')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, 20.0, (cam.img_width, cam.img_height))

    image_width = cam.img_width
    image_height = cam.img_height

    # Assuming the camera's field of view (FOV)
    fov_x = float(os.getenv('FOV_X', 90))  # degrees
    fov_y = float(os.getenv('FOV_Y', 60))  # degrees

    # Initialize SORT tracker
    tracker = Sort()

    # Initialize dictionaries to track unique IDs and heading angles
    object_tracker = {}  # Map to track heading angles by unique object IDs

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break

        img = cam.read()
        if img is None:
            break
        
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        # Filter detections
        filtered_boxes = []
        filtered_confs = []
        filtered_clss = []

        for box, conf, cls in zip(boxes, confs, clss):
            if cls in allowed_class_ids and conf >= DETECTION_ACCURACY_THRESHOLD:
                filtered_boxes.append(box)
                filtered_confs.append(conf)
                filtered_clss.append(cls)

        # Convert boxes to format suitable for SORT (x_min, y_min, x_max, y_max)
        detections = np.array(filtered_boxes, dtype=np.float32)

        if len(detections) == 0:
            print("No detections found, skipping tracking for this frame.")
            continue  # Skip to the next frame if no detections are found

        # Update SORT tracker with current detections
        tracked_objects = tracker.update(detections)

        # Save cropped images of detections
        save_cropped_images(img, filtered_boxes, filtered_clss, filtered_confs, cls_dict)

        # Track heading angles and report detections
        for obj in tracked_objects:
            x_min, y_min, x_max, y_max, track_id = obj.astype(int)

            # Get class index
            class_id = int(filtered_clss[0]) if filtered_clss else -1
            class_name = cls_dict.get(class_id, 'Unknown')

            # Calculate heading angles
            theta_x, theta_y = calculate_heading((x_min, y_min, x_max, y_max), image_width, image_height, fov_x, fov_y)

            # Ensure the object is reported
            if track_id not in object_tracker:
                # New object detected, report it
                object_tracker[track_id] = (theta_x, theta_y)
                print(f"Detected {class_name} {track_id} at Heading angle (horizontal): {theta_x:.2f} degrees, (vertical): {theta_y:.2f} degrees")
            else:
                # Existing object, check if there's any significant change in heading
                prev_theta_x, prev_theta_y = object_tracker[track_id]
                if abs(theta_x - prev_theta_x) > 0.1 or abs(theta_y - prev_theta_y) > 0.1:
                    print(f"Updated {class_name} {track_id} at Heading angle (horizontal): {theta_x:.2f} degrees, (vertical): {theta_y:.2f} degrees")
                    object_tracker[track_id] = (theta_x, theta_y)

            # Draw the bounding box and the object ID
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, f'{class_name} {track_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img = vis.draw_bboxes(img, filtered_boxes, filtered_confs, filtered_clss)
        img = show_fps(img, fps_calculator.update())
        cv2.imshow(WINDOW_NAME, img)
        
        # Write frame to video
        video_writer.write(img)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('q'):  # Press 'q' to quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

    # Release video writer
    video_writer.release()


def main():
    global OUTPUT_DIR  # Ensure we can modify the global OUTPUT_DIR
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    
    cam = Camera(args.rtsp_url)
      # Initialize Camera from utils.rtspcam
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    # Create a new experiment directory
    OUTPUT_DIR = create_new_experiment_dir(BASE_OUTPUT_DIR)

    # Specify allowed classes directly in the code
    allowed_classes = None  # Set to None to detect all classes, or specify a list of classes.

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis, allowed_classes, cls_dict)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

