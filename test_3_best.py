import cv2
import torch
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import os

num = 0

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
        x = self.relu(self.fc2(x)) 
        return x
    


model = SimpleCNN(num_classes=2) 
model.load_state_dict(torch.load('/home/hankla/Desktop/work/puddle_deploy/yolov8_models/classi_puddle.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

# transform = transforms.Compose([
#     transforms.ToPILImage(),  
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((200, 200)),
#     transforms.ToTensor()])

yolo_model = YOLO("/home/hankla/Desktop/work/puddle_deploy/yolov8_models/segment_puddle.pt")

# idx = 2


# name_path = os.listdir('/home/hankla/Desktop/work/puddle_deploy/data_test/2024-01-10/thermal')
# video_path = '/home/hankla/Desktop/work/puddle_deploy/data_test/2024-01-10/thermal/' + str(name_path[idx])

# vid train new
video_path = '/home/hankla/Desktop/work/puddle_deploy/data_test/frame_thermal_095003 (online-video-cutter.com).mp4'

# #vidtrain past
# video_path = '/home/hankla/Desktop/work/puddle/dataset_VDO/2023-12-26_puddle/thermal/all.mp4'


# # "none All"
# video_path = '/home/hankla/Desktop/work/puddle/dataset_VDO/2023-12-25_none_puddle/Thermal record 2023-08-15 13_51_36.mp4'

cap = cv2.VideoCapture(video_path)

num = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.merge([img, img, img])

    # img = cv2.resize(img, (1280, 720))
    # x, y, w, h = 106, 59, 1223 - 106, 686 - 64
    # img = img[y:y + h, x:x + w]


    h, w, _ = img.shape
    results = yolo_model.predict(source=img,
                            save=None,
                            save_txt=False,
                            device='cpu',
                            verbose=False,
                            conf=0.2,
                            iou=0.1,
                            classes=[0],
                            imgsz=(640, 640))
    
    for r in results:
        boxes = r.boxes
        masks = r.masks
        probs = r.probs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                xmin = int(box.data[0][0])
                ymin = int(box.data[0][1])
                xmax = int(box.data[0][2])
                ymax = int(box.data[0][3])

                point_detec = xmin,ymin,xmax,ymax

                plus_area = 10

                xmin -= plus_area 
                xmax += plus_area 
                ymin -= plus_area +10
                ymax += plus_area +10

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)

                roi = img[ymin:ymax, xmin:xmax]
                cv2.imshow('roi', roi)
                roi = cv2.resize(roi,(200,200))
                roi_save = roi.copy()
 

                # cv2.imshow('roi', roi)

                roi = Image.fromarray(roi)  
                roi = transform(roi)
                # roi = roi.unsqueeze(0) 
                
                # print(roi.shape)

                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                confidence_text = str(np.around(box.conf,3))
                
                with torch.no_grad():

                    output = model(roi)
                    _, predicted_class = torch.max(output, 1)
                    
                    value = predicted_class.item()
                    confidence_values = torch.softmax(output, dim=1).squeeze().tolist()
                    
                    if value == 1:
                        if confidence_values[1]   >= 0.8 : #and confidence_values[1] > confidence_values[0] 
                            # print(predicted_class[0] ,confidence_values[1])

                            print(confidence_values, num)
                            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            # cv2.putText(img, str(np.around((confidence_values[1]),5)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)



                cv2.rectangle(img, (point_detec[0], point_detec[1]), (point_detec[2], point_detec[3]), (0, 0, 255), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite('/home/hankla/Desktop/work/puddle_deploy/dataimage_seg/'+str(num)+'P' + '.jpg', img)
        num += 1
        # time.sleep(1)
        

cap.release()
cv2.destroyAllWindows() 