import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# idx = 13
# name_path = os.listdir('/home/hankla/Desktop/work/puddle_deploy/data_test/2024-01-10/thermal')
# video_path = '/home/hankla/Desktop/work/puddle_deploy/data_test/2024-01-10/thermal/' + str(name_path[idx])


mode = 0

# if mode == 0:  #puddle
#     video_path = '/home/hankla/Desktop/work/puddle_deploy/data_test/frame_thermal_095003 (online-video-cutter.com).mp4'
# if mode == 1: #None puddle
#     video_path = '/home/hankla/Desktop/work/puddle/dataset_VDO/2023-12-25_none_puddle/Thermal record 2023-08-15 13_51_36.mp4'
idx = 0
name_path = os.listdir('/home/hankla/Desktop/work/puddle_deploy/Data_test11_03_67/2024-03-07/thermal')
video_path = '/home/hankla/Desktop/work/puddle_deploy/Data_test11_03_67/2024-03-07/thermal/' + str(name_path[idx])



class Yolov8Segmentation():
    def __init__(self):
        self.device_default = 0
        self.gpu_check_point()
        self.load_ai_image_segmentation()

    def load_ai_image_segmentation(self):
        import os
        from pathlib import Path
        def combine_f(this_root, finfo):
            return os.path.join(this_root, finfo)
        this_file = Path(__file__).resolve()
        model_path = combine_f(combine_f(this_root=this_file.parents[0], finfo="yolov8_models"), finfo="/home/hankla/Desktop/work/puddle_deploy/yolov8_models/segment_puddle.pt")
        self.yolov8_model = YOLO(model_path)


    def gpu_check_point(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = self.device_default
        print(f"Device : {self.device}")

    def detect(self, img):
        ret_pkg = dict(), list(), list(), list(), list(), list()
        height, width, channels = img.shape
        results = self.yolov8_model.predict(source=img, 
                                        save=False, 
                                        save_txt=False, 
                                        device=self.device, 
                                        max_det=10, 
                                        verbose=False,
                                        conf=0.8,
                                        iou=0.2,
                                        classes=[0])
        names, class_ids, seg_objects, scores, obbs, bbs = ret_pkg
        for result in results:
            names = result.names
            if result.masks is None: return ret_pkg
            masks_segs = result.masks.segments
            seg_object, seg2obbs, seg2bbs = [], [], []
            for seg in masks_segs:
                seg[:, 0] *= width
                seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                seg_object.append(segment)
                seg2obbs.append(self.get_rotated_rectangle(segment=segment))
                seg2bbs.append(cv2.boundingRect(segment))
            class_id = np.array(result.boxes.cls.cpu(), dtype="int")
            score = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
            class_ids.append(class_id)
            seg_objects.append(seg_object)
            obbs.append(seg2obbs)
            bbs.append(seg2bbs)
            scores.append(score)
        return names, list(class_ids[0]), list(seg_objects[0]), list(scores[0]), list(obbs[0]), list(bbs[0])
    
    def get_rotated_rectangle(self, segment):
        rect = cv2.minAreaRect(segment)
        x_obb, y_obb, w_tp_obb, h_tp_obb, a_obb = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        w_tp_obb, h_tp_obb, a_obb = abs(w_tp_obb), abs(h_tp_obb), a_obb % 180
        w_tp_obb, h_tp_obb = self.extend_size_obb(w_tp_obb, h_tp_obb)
        return x_obb, y_obb, w_tp_obb, h_tp_obb, a_obb
    
    def extend_size_obb(self, w_tp_obb, h_tp_obb):
        self.zpctx = 110 # Padding x
        self.zpcty = 200 # Padding x
        zpctw, zpcth = self.zpctx * 0.01, self.zpcty * 0.01
        if (w_tp_obb > h_tp_obb): w_tp_obb, h_tp_obb = w_tp_obb * zpctw, h_tp_obb * zpcth
        else: w_tp_obb, h_tp_obb = w_tp_obb * zpcth, h_tp_obb * zpctw
        return w_tp_obb, h_tp_obb
    
    def crop_rotated_rect(self, image, seg_object, bb):
        roi_size = 200 #### ROI Size
        x, y, w, h = bb
        center = (int(x + w / 2), int(y + h / 2))
        roi = cv2.getRectSubPix(image, (int(w * self.zpctx * 0.01), int(h * self.zpcty * 0.01)), center)
        if not len(seg_object): roi = np.zeros((roi_size,roi_size,3), np.uint8)
        return roi
    
    def get_roi(self, img):
        rois, img_copy = [], img.copy()
        names, class_ids, seg_objects, scores, obbs, bbs = self.detect(img=img)
        for i in range(len(class_ids)):
            cx, cy, hw, hh, angle = obbs[i]
            box_points = cv2.boxPoints(((cx, cy), (hw , hh), angle)).astype(np.int0)
            cv2.polylines(img, [box_points], True, (255, 255, 255), 1)
            cv2.polylines(img, [seg_objects[i]], True, (0, 255, 0), 1)
            rois.append(self.crop_rotated_rect(image=img_copy, seg_object=seg_objects[i], bb=bbs[i]))
        return names, class_ids, seg_objects, scores, obbs, img, rois

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 25 * 25, 16)
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = self.relu(self.fc1(x))
        model_output = self.relu(self.fc2(x))  
        return model_output

class DNNOperator():
    def __init__(self, model_path, num_classes):
        self.tran = transforms.Compose([ 
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((200, 200)),
            transforms.ToTensor()])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_load = SimpleCNN(num_classes=num_classes)
        self.load_model(model_path=model_path)

    def load_model(self, model_path):
        self.model_load.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model_load.to(self.device)
        self.model_load.eval()

    def predict(self,roi):
        with torch.no_grad():
            dict_confident = dict()
            roi = Image.fromarray(roi)
            roi_tensor = self.tran(roi).unsqueeze(0)
            output = self.model_load(roi_tensor)
            confidence_values = torch.softmax(output, dim=1).squeeze().tolist()
            for i, item in enumerate(confidence_values):
                dict_confident[i] = item
            return dict_confident


yolov8_segmentation : Yolov8Segmentation = Yolov8Segmentation()

model_classify = DNNOperator(num_classes=2 ,model_path='/home/hankla/Desktop/work/puddle_deploy/yolov8_models/classi_puddle.pth' )

vid = cv2.VideoCapture(video_path) 
  
count = 0

while(True):
    ret, frame = vid.read() 
    if mode == 1:
        frame = cv2.resize(frame, (1280, 720))
        x, y, w, h = 106, 59, 1223 - 106, 686 - 64
        frame = frame[y:y + h, x:x + w]

    if not ret:
        print('Done')
        break

    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    names, class_ids, seg_objects, scores, boxs, img, rois = yolov8_segmentation.get_roi(img=gray)


    if rois != []:
        cv2.imshow('rois',rois[0])
    for roi in rois:
        predicted_result = model_classify.predict(roi)
        print(predicted_result)

    # print(predicted_result)


    cv2.imshow('frame', img) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(count) 
        break

vid.release()
cv2.destroyAllWindows() 


