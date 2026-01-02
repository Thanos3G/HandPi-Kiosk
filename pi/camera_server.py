# camera_server.py
# Publishes frames via shared memory for MediaPipe 

import time
import numpy as np
import cv2
from multiprocessing import shared_memory

WIDTH = 640
HEIGHT = 480
SHM_NAME = "handpi_frames"

def open_camera():
    # USB first 
    for i in range(13):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            print(f"[camera_server] Using USB camera index {i}")
            return ("v4l2", cap)

    # Fallback to CSI
    from picamera2 import Picamera2
    picam = Picamera2()
    picam.configure(
        picam.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
        )
    )
    picam.start()
    print("[camera_server] Using CSI camera via Picamera2")
    return ("csi", picam)

def main():
    size = WIDTH * HEIGHT * 3
    shm = shared_memory.SharedMemory(create=True, size=size, name=SHM_NAME)
    frame_buf = np.ndarray((HEIGHT, WIDTH, 3), dtype=np.uint8, buffer=shm.buf)

    backend, cam = open_camera()

    try:
        while True:
            if backend == "v4l2":
                ok, frame = cam.read()
                if not ok:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cam.capture_array()

            frame_buf[:] = frame
            time.sleep(1 / 30)
    finally:
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    main()
