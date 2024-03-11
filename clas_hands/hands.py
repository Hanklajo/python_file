import cv2
import mediapipe as mp
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python


class HandsDetection:
    def __init__(self, modelpath, is_cuda=True):
        self.config = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        if is_cuda:
            base_options = python.BaseOptions(model_asset_path=modelpath, delegate=python.BaseOptions.Delegate.GPU)
        else:
            base_options = python.BaseOptions(model_asset_path=modelpath)
        
        self.options = HandLandmarkerOptions(base_options=base_options) 
        self.landmarker = self.HandLandmarker.create_from_options(self.options)
        

        # self.hands = self.landmarker.Hands(
        #     static_image_mode=True,
        #     max_num_hands=5,
        #     min_detection_confidence=0.8,
        #     min_tracking_confidence=0.2,
        #     base_options=base_options)
        self.depth_threshold = 0.001

    def process_image(self, frame):
        img_mp = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.landmarker.detect(mp.Image(image_format= mp.ImageFormat.SRGB, data=img_mp))
        dets = []
        landmark = []

        for face_landmark in results.hand_landmarks:
            for mask in face_landmark:
                landmark.append({'x': mask.x, 'y': mask.y, 'z': mask.z})
            break

        if landmark:
            maps_xyz = []
            for landmarks_per_hand in results.hand_landmarks:
                if all(0 <= lm.x <= 1 and 0 <= lm.y <= 1 for lm in landmarks_per_hand):
                    lx = [landmark.x for landmark in landmarks_per_hand]
                    ly = [landmark.y for landmark in landmarks_per_hand]  
                    lz = [landmark.z for landmark in landmarks_per_hand]
                    finger_z = landmarks_per_hand[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].z
                    finger_y = int(landmarks_per_hand[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * rgb_frame.shape[0])
                    wrist_y = int(landmarks_per_hand[self.mp_hands.HandLandmark.WRIST].y * rgb_frame.shape[0])
                    wrist_z = landmarks_per_hand[self.mp_hands.HandLandmark.WRIST].z
                    map_xyz = list(zip(lx, ly, lz))
                    maps_xyz.append([map_xyz, finger_z, finger_y, wrist_y, wrist_z])

            for map_xyz in maps_xyz:
                landmarks, finger_z, finger_y, wrist_y, wrist_z = map_xyz
                for ld in landmarks:
                    x, y = int(ld[0] * frame.shape[1]), int(ld[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                landmarks_x, landmarks_y, landmarks_z = zip(*landmarks)
                landmarks_x = list(landmarks_x)
                landmarks_y = list(landmarks_y)
                landmarks_z = list(landmarks_z)
                print(landmarks_x)

                # if finger_y < wrist_y and finger_z < wrist_z - self.depth_threshold:
                #     det = self.dnn_operator.dnn_classification(predict_input=landmarks_x+landmarks_y,
                #                                                 output_size=6, 
                #                                                 output_map=self.hand_classification)
                    # if det[1] >= 0.975:
                    #     dets.append(det[0])
                    #     cv2.putText(frame, f'{det[0]}: {det[1]:.2f}', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    #     break
        return frame, str() if (len(dets) == 0) else dets[0]

model_hands = HandsDetection(modelpath='/home/hankla/Desktop/work/clas_hands/hand_landmarker.task',is_cuda=False)
cap = cv2.VideoCapture(0)

while True:
    _ , frame = cap.read()
    frame_det , clas = model_hands.process_image(frame)
    cv2.imshow('framedet', frame_det)
    print(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
#pp