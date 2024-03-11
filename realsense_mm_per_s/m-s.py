import math
import rospy
import numpy as np
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import time
  
class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.im_combi0 = rospy.Publisher('im_combi0', Image)
        self.im_combi1 = rospy.Publisher('im_combi2', Image)

    def publisher_image(self, image_rgb0, image_rgb1):
        combined_frame0 = self.bridge.cv2_to_imgmsg(image_rgb0, "passthrough")
        combined_frame1 = self.bridge.cv2_to_imgmsg(image_rgb1, "passthrough")
        self.im_combi0.publish(combined_frame0)
        self.im_combi1.publish(combined_frame1)

    def convert_scale(self, img1, img2):
        cv_image1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding="passthrough")
        cv_image2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding="passthrough")
        return cv_image1, cv_image2

    def get_pararos(self, info1, info2):
        P1 = info1.P[2], info1.P[5]
        P2 = info2.P[2], info1.P[5]
        return P1, P2


class SpeedCalculator:
    _G = 9.81
    @staticmethod
    def cal_speed(times):
        g = SpeedCalculator._G
        s = 0.165 
        t = times  
        u = (s - 0.5 * g * (t**2)) / t
        return SpeedCalculator.__calspeed2(u, times)
    @staticmethod
    def __calspeed2(speed1, times):
        g = SpeedCalculator._G
        v2 = g * times + speed1
        return SpeedCalculator.__time_s(v2)
    @staticmethod
    def __time_s(speed2):
        g = SpeedCalculator._G
        s = 0.1
        sqr = math.sqrt((speed2**2 + 2 * g * s))
        t2 = (-speed2 - sqr) / g 
        vs = g * t2 + speed2
        print(t2)
        return abs(t2), abs(vs)



def callback(ros_img1, ros_img2 ,info1,info2):
    global fall_start_time, time_buffer
    dis_cam2obj_mm = 400 

    try:
        cv_image1 , cv_image2 =  processor.convert_scale(ros_img1,ros_img2)
        principal_point1,principal_point2 = processor.get_pararos(info1,info2) 
        roi_1 = cv_image1[:, int(principal_point1[0])][cv_image1[:, int(principal_point1[0])] != 0]
        roi_2 = cv_image2[:, int(principal_point2[0])][cv_image2[:, int(principal_point2[0])] != 0]
        if np.min(roi_2) < dis_cam2obj_mm  :
            if fall_start_time is None:
                fall_start_time = time.time()
                print('timestart', np.min(roi_2))

        else:
            if fall_start_time is not None: 
                if np.min(roi_1) < dis_cam2obj_mm  :
                    fall_end_time = time.time()
                    fall_duration = fall_end_time - fall_start_time
                    print("Time taken for object to fall through center line: {:.6f} seconds".format(fall_duration))
                    print('timestop')
                    t2 , vs = t2, vs = SpeedCalculator.cal_speed(fall_duration)
                    print(t2 , 'sec --- speed' , vs *1000, 'mm/s')
                    fall_start_time = None
        processor.publisher_image(cv_image1 , cv_image2)

    except Exception as e:
        print(e)        
        


if __name__ == '__main__':       
    fall_start_time = None
    time_buffer = []
    rospy.init_node('infrared_subscriber') 
    sub_rgb1 = message_filters.Subscriber("/cam_1/depth/image_rect_raw", Image)
    sub_rgb2 = message_filters.Subscriber("/cam_2/depth/image_rect_raw", Image)
    info_depth_sub1 = message_filters.Subscriber('/cam_1/depth/camera_info', CameraInfo)
    info_depth_sub2 = message_filters.Subscriber('/cam_2/depth/camera_info', CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer([sub_rgb1, sub_rgb2 , info_depth_sub1, info_depth_sub2 ], 10, 0.1, allow_headerless=True)
    processor = ImageProcessor()

    ts.registerCallback(callback)
    rospy.spin()




        # time_buffer.append(t0)
        # if time_buffer.__len__() == 2:
        #     tp = abs(time_buffer[1] - time_buffer[0])
        #     fps = 1 / tp
        #     print(fps)
        #     time_buffer.clear()

        # tp = time.time() - t0
        # fps = 1 / tp
        # print(fps)  # Print FPS