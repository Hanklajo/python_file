import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 90)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)  # Enable depth stream

# Start streaming
pipeline.start(config)

# Define colormap for heatmap
colormap = cv2.COLORMAP_JET

# Global variable to store time when object starts falling
fall_start_time = None



def show_colormap(depth_image):
    global fall_start_time

    depth_image = np.uint16(depth_image)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), colormap)

    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)

 

    x_center = depth_colormap.shape[1] // 2  # หาตำแหน่ง x ที่อยู่ตรงกลางของภาพ
    cropped_image = depth_colormap[240:260, x_center] 
    

    cv2.imshow('Cropped Depth Image', cropped_image)
    # cv2.line(ir_image, (ir_image.shape[1] // 2, 0), (ir_image.shape[1] // 2, ir_image.shape[0]), (255, 255, 255), 2)
    # cv2.line(depth_colormap, (depth_colormap.shape[1] // 2, 0), (depth_colormap.shape[1] // 2, depth_colormap.shape[0]), (255, 255, 255), 2)
    cv2.imshow('depth_colormap', depth_colormap)

    # print(np.min(cropped_image))
    # Check if object is falling through the center line
    if np.min(cropped_image) < 20  :
        if fall_start_time is None:
            fall_start_time = time.time()
            print('timestart',np.min(cropped_image))
    else:
        if fall_start_time is not None:
            fall_end_time = time.time()
            fall_duration = fall_end_time - fall_start_time
            print("Time taken for object to fall through center line: {:.2f} seconds".format(fall_duration))
            print('timestop')
            fall_start_time = None



try:
    while True:
        t0 = time.time()
        # Wait for a coherent pair of frames: IR frame and depth frame
        frames = pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame()
        depth_frame = frames.get_depth_frame()

        if ir_frame and depth_frame:
            ir_image = np.asanyarray(ir_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            show_colormap(depth_image.copy())
            cv2.imshow('ir_image', ir_image)

        tp = time.time() - t0
        fps = 1 / tp
        # print(fps)  # Print FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
