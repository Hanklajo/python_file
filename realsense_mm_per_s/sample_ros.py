#!/usr/bin/python3
import tf
import cv2
import math
import rospy
import torch
import numpy as np
import message_filters
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo


class YolovSegmentation():
    '''
    Start Yolov8 OBB.
    '''
    def __init__(self):
        self.device_default = 0
        self.gpu_check_point()
        self.load_ai_image_segmentation()

    def load_ai_image_segmentation(self):
        import os
        from pathlib import Path
        def combine_f(this_root, finfo):
            return os.path.join(this_root, finfo)
        this_file = Path(_file_).resolve()
        model_path = combine_f(combine_f(this_root=this_file.parents[0], finfo="yolov8_models"), finfo="yolov8n-seg.pt")
        self.yolov8_model = YOLO(model_path)

    def gpu_check_point(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = self.device_default
        print(f"Device : {self.device}")

    def detect(self, img):
        height, width, channels = img.shape
        results = self.yolov8_model.predict(source=img, 
                                     save=False, 
                                     save_txt=False, 
                                     device=0,
                                     verbose=False,
                                     conf=0.55,
                                     iou=0.2,
                                     classes=[0])   
        bboxes, class_ids, seg_idxs, scores = [], [], [], []
        names = {}
        for result in results:
            names = result.names
            seg_idx = []
            try:
                _segs = result.masks.segments
            except:
                return {}, [], [], []      
            for seg in _segs:
                seg[:, 0] *= width
                seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                seg_idx.append(segment)
            bboxe = np.array(result.boxes.xyxy.cpu(), dtype="int")
            class_id = np.array(result.boxes.cls.cpu(), dtype="int")
            score = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
            bboxes.append(bboxe)
            class_ids.append(class_id)
            seg_idxs.append(seg_idx)
            scores.append(score)
        return names, list(class_ids[0]), list(seg_idxs[0]), list(scores[0])
    
    def get_seg(self, img):
        names, class_ids, seg_idxs, scores = self.detect(img)
        for i in range(len(class_ids)):
            cv2.polylines(img, [seg_idxs[i]], True, (0, 255, 0), 2)
        return img, seg_idxs, names, class_ids

class DepthImageLaserScan(YolovSegmentation):
    ENABLE_AI = True
    ROOT_NAME = 'detect'
    ROS_PARAM_TIMEOUT = 1000
    def __init__(self):
        rospy.init_node('depth_image_laser_scan_node')
        if self.ENABLE_AI: super().__init__() 
        self.bridge = CvBridge()
        # Broadcaster object frames
        self.broadcaster_object_frames = tf.TransformBroadcaster()
        # Synchronized Subscribers
        info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        info_depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        # ApproximateTimeSynchronizer or TimeSynchronizer can be used based on the requirement
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, info_sub, depth_sub, info_depth_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)
        # Publishers
        self.color_msg_pub = rospy.Publisher(f'/camera/{self.ROOT_NAME}/color/image_raw', Image, queue_size=10)
        self.depth_msg_pub = rospy.Publisher(f'/camera/{self.ROOT_NAME}/aligned_depth_to_color/image_raw', Image, queue_size=10)
        self.info_depth_pub = rospy.Publisher(f'/camera/{self.ROOT_NAME}/aligned_depth_to_color/camera_info', CameraInfo, queue_size=10)
        self.latest_camera_info = None

    def callback(self, color_msg: Image, camera_info_msg: CameraInfo, depth_msg: Image, depth_info_msg : CameraInfo):
        # self.color_msg
        # self.camera_info_msg
        # self.depth_msg
        # self.depth_info_msg
        self.send_laser_scan_from_depth_image_tf(frame_id=camera_info_msg.header.frame_id)
        current_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        current_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if self.ENABLE_AI:
            current_color_tmp, seg, names, class_ids = self.get_seg(img=current_color.copy())
            filtered_depth, current_color_tmp = self.apply_segmentation_to_depth(current_depth, 
                                                                                seg, 
                                                                                current_color_tmp, 
                                                                                names, 
                                                                                class_ids, 
                                                                                camera_info_msg)
        else:
            filtered_depth, current_color_tmp = current_depth, current_color.copy()
        seg_color = self.drf(image=current_color_tmp, cam_info=camera_info_msg)
        seg_color_msg = self.bridge.cv2_to_imgmsg(seg_color, encoding="bgr8")
        filtered_depth_msg = self.bridge.cv2_to_imgmsg(filtered_depth, encoding="passthrough")
        # Set appear to be synchronized topics
        filtered_depth_msg.header = depth_msg.header
        depth_info_msg.header = depth_msg.header
        # Publish topics
        self.color_msg_pub.publish(seg_color_msg)
        self.depth_msg_pub.publish(filtered_depth_msg)
        self.info_depth_pub.publish(depth_info_msg)

    def send_laser_scan_from_depth_image_tf(self, frame_id):
        self.broadcaster_object_frames.sendTransform((0.0,0.0,0.0),
                                                     tf.transformations.quaternion_from_euler(0.0, -1.5708, 1.5708),
                                                     rospy.Time.now(),
                                                     "laser_scan_from_depth_image",
                                                     frame_id)

    def apply_segmentation_to_depth(self, depth_img, seg, image, names, class_ids, cam_info : CameraInfo):
        def calculate_the_moments_of_the_polygon(s):
            cX, cY = 0,0
            M = cv2.moments(s)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            return cX, cY
        def euler_to_rotation_matrix(rx, ry, rz):
            rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
            cos_rx, sin_rx = np.cos(rx), np.sin(rx)
            cos_ry, sin_ry = np.cos(ry), np.sin(ry)
            cos_rz, sin_rz = np.cos(rz), np.sin(rz)
            rotation_matrix = np.array([
                [cos_ry * cos_rz, cos_rz * sin_rx * sin_ry - cos_rx * sin_rz, cos_rx * cos_rz * sin_ry + sin_rx * sin_rz],
                [cos_ry * sin_rz, cos_rx * cos_rz + sin_rx * sin_ry * sin_rz, -cos_rz * sin_rx + cos_rx * sin_ry * sin_rz],
                [-sin_ry, cos_ry * sin_rx, cos_rx * cos_ry]
            ])
            return rotation_matrix
        centroids = []
        mask = np.zeros(depth_img.shape, dtype=np.uint8)
        for i, s in enumerate(seg):
            cv2.fillPoly(mask, [s], 1)
            cx, cy = calculate_the_moments_of_the_polygon(s=s)
            ux, uy, z = self.deprojection_from_depth(cX=cx, cY=cy, depth=depth_img, cam_info=cam_info)
            tvec = np.array([ux, uy, z]).reshape(3, 1).reshape(3, 1)
            rvec = euler_to_rotation_matrix(rx=0.0, ry=0.0, rz=0.0)
            image = cv2.circle(image, (cx, cy), 7, (255,255,255), -1)
            image = cv2.drawFrameAxes(image, np.array(cam_info.K).reshape(3, 3), np.array([list(cam_info.D)]), rvec, tvec, 0.07)
            image = cv2.putText(image, names[class_ids[i]], (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_d, y_d = z, -1.0 * ux
            centroids.append((names[class_ids[i]], x_d, y_d, 0.0, 0.0, 0.0, 0.0))
        filtered_depth = np.where(mask == 1, depth_img, 0)
        self.laser_scan_broadcast(centroids=centroids)
        return filtered_depth, image
    
    def laser_scan_broadcast(self, centroids):
        def realign_list(input_list):
            input_sorted = sorted(input_list, key=lambda x: x[1], reverse=False)
            realigned_list = []
            count_dict = {}
            for item in input_sorted:
                label = item[0]
                count_dict.setdefault(label, 0)
                new_label = f"{label}{count_dict[label]}"
                realigned_list.append((new_label,) + item[1:])
                count_dict[label] += 1
            return realigned_list
        def deg_to_rad(deg):
            return deg * math.pi / 180.0
        realign_centroids = realign_list(input_list=centroids)
        for centroid in realign_centroids:
            trans_det = (centroid[1], centroid[2], centroid[3])
            quaternion_det = tf.transformations.quaternion_from_euler(deg_to_rad(centroid[4]), 
                                                                      deg_to_rad(centroid[5]), 
                                                                      deg_to_rad(centroid[6]))
            self.broadcaster_object_frames.sendTransform(trans_det,
                                                         quaternion_det,
                                                         rospy.Time.now(),
                                                         centroid[0],
                                                         "laser_scan_from_depth_image")

    def drf(self, image, cam_info : CameraInfo):
        MK = cam_info.K
        SH = self.wnp(param_name='/depthimage_to_laserscan/scan_height', param_value=1)
        image_tmp = image.copy()
        cv2.line(image_tmp,(0, int(MK[5])),(int(MK[2] * 2), int(MK[5])),(255,255,255), SH)
        image = cv2.addWeighted(image_tmp, 0.5, image, 1 - 0.5, 0)
        cv2.arrowedLine(image, (int(MK[2]), int(MK[5])), (int(MK[2]), int(MK[5]) + 50), (0, 230, 0), 2, tipLength = 0.2)
        cv2.arrowedLine(image, (int(MK[2]), int(MK[5])), (int(MK[2]) + 50, int(MK[5])), (0, 0, 230), 2, tipLength = 0.2)
        return image
    
    def wnp(self, param_name, param_value):
        for i in range(self.ROS_PARAM_TIMEOUT):
            if rospy.has_param(param_name):
                param_value = rospy.get_param(param_name)
                break
        return param_value
    
    def deprojection_from_depth(self, cX, cY, depth, cam_info : CameraInfo):
        ## len(cam_info.D) >= 5
        intr = cam_info.K
        coeffs = cam_info.D
        z = float(depth[cY, cX]) * 0.001
        x = z * (cX - intr[2]) / intr[0]
        y = z * (cY - intr[5]) / intr[4]
        r2 = x*x + y*y
        f = 1 + coeffs[0]*r2 + coeffs[1]*r2*r2 + coeffs[4]*r2*r2*r2
        ux = x*f + 2*coeffs[2]*x*y + coeffs[3]*(r2 + 2*x*x)
        uy = y*f + 2*coeffs[3]*x*y + coeffs[2]*(r2 + 2*y*y)
        return ux, uy, z


    def run(self):
        rospy.spin()

if _name_ == '_main_':
    node = DepthImageLaserScan()
    node.run()