import cv2

import cv2

class Camera():
    """Camera class for handling RTSP streams only using cv2."""

    def __init__(self, rtsp_url, width=640, height=480):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(rtsp_url)
        
        # Set the resolution if possible
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.is_opened = self.cap.isOpened()
        
        # Initialize width and height for later use
        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def isOpened(self):
        return self.is_opened

    def read(self):
        """Read a frame from the RTSP stream."""
        if not self.is_opened:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """Release the video capture object."""
        if self.cap:
            self.cap.release()
        self.is_opened = False

    def __del__(self):
        self.release()

