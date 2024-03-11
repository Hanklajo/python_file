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

def show_colormap(depth_image):
    depth_image = np.uint8(depth_image.astype(float) / 256)
    depth_colormap = cv2.applyColorMap(depth_image, colormap)

    cv2.line(ir_image, (0, ir_image.shape[0] // 2), (ir_image.shape[1], ir_image.shape[0] // 2), (255, 255, 255), 2)
    cv2.line(depth_colormap, (0, depth_colormap.shape[0] // 2), (depth_colormap.shape[1], depth_colormap.shape[0] // 2), (255, 255, 255), 2)
    cv2.imshow('Depth Heatmap', depth_colormap)

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
        print(fps)  # Print FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
