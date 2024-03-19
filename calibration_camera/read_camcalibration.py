import cv2
import numpy as np

class CameraUndistort():
    def __init__(self, image_shape): 
        self.ck = np.array([[656.204124160109, 0,325.456291893977],
                        [0,655.734140548410,212.024046302360],
                        [0, 0, 1]])

        self.cd = np.array([[-0.359119196514515,0.0871241163651550, 0.0000890758039139583 ,0.000372151398277607 ,0.0613257291162202]])   #radialsistortion-
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.ck, self.cd, (image_shape[0], image_shape[1]), 0)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.ck, self.cd, None, self.new_camera_matrix, (image_shape[0], image_shape[1]), 5)
        self.color_info = {"K": self.ck, "D": self.cd}

    def undistort_image(self, image):
        undistorted_image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]
        print(undistorted_image.shape)
        return undistorted_image

def find_corners_size_in_pixel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    chessboard_size = (9,6)  
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        square_size_pixels = abs(corners[1, 0, 0] - corners[0, 0, 0])
        square_size_mm = 20
        mm_per_pixel = square_size_mm / square_size_pixels
        print("mm per pixel:", mm_per_pixel)
    else:
        print("Corners not found!")

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

camera_undistort : CameraUndistort = CameraUndistort(image_shape=[640,480])
while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break
    undistorted_img = camera_undistort.undistort_image(image=frame.copy())
    mm_per_pixel = find_corners_size_in_pixel(image=undistorted_img.copy())
    cv2.imshow('Camera', frame)
    cv2.imshow('Undistorted Image', undistorted_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
