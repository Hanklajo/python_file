import os
import cv2
import yaml
import mediapipe as mp
from pathlib import Path


import pandas as pd
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
import random

class SimpleDNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleDNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

    def forward(self, model_input):
        dnnAsrelu1 = torch.nn.functional.relu(self.fc1(model_input))
        dnnAsrelu2 = torch.nn.functional.relu(self.fc2(dnnAsrelu1))
        model_output = self.fc3(dnnAsrelu2)
        return model_output

class DNNOperator():
    def __init__(self):
        self.model_load, self.is_init = None, True
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_style
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=5, min_detection_confidence=0.9,
                       min_tracking_confidence = 0.2)
        self.depth_threshold = 0.03
        self.load_config()

    def load_config(self):
        def combine_f(this_root, finfo):
            return os.path.join(this_root, finfo)
        this_file = Path(__file__).resolve() 
        this_root = this_file.parents[1]
        self.this_config = combine_f(combine_f(this_root=this_root, finfo="models"), finfo="model_weights.pth")

    def load_dataset(self, file_name, head_name):
        data = pd.read_csv(file_name)
        df = pd.DataFrame(data)
        self.landmarks_list = df.drop(columns=[head_name]).values.tolist()
        self.classes_list = df[head_name].tolist()
        self.input_size = len(self.landmarks_list[0])
        self.output_size = len(df[head_name].unique().tolist())
        self.x_train_tensor = torch.tensor(self.landmarks_list, dtype=torch.float32)  
        self.y_train_tensor = torch.tensor(self.classes_list, dtype=torch.int64)

    def train_dnn(self, batch_size=6, lr=0.001, num_epochs=1000):
        model = SimpleDNN(input_size=self.input_size, output_size=self.output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)  
        train_dataset = TensorDataset(self.x_train_tensor, self.y_train_tensor)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        torch.save(model.state_dict(), self.this_config)

    def dnn_classification(self, predict_input, output_size, output_map):
        predicted_class_map, predicted_probability = str(), 0.0
        if self.is_init:
            self.model_load = SimpleDNN(input_size=len(predict_input), output_size=output_size)
            self.model_load.load_state_dict(torch.load(self.this_config))
            self.model_load.eval()
            self.is_init = False
        with torch.no_grad():
            sample_input = torch.tensor(predict_input, dtype=torch.float32).unsqueeze(0)
            logits = self.model_load(sample_input)  
            probabilities = F.softmax(logits, dim=1)  
            _, predicted_class = torch.max(probabilities, 1)
            predicted_class_map = output_map[int(predicted_class[0].item())]  
            predicted_probability = probabilities[0, predicted_class].item()
        return predicted_class_map, predicted_probability


class HandsDetection:
    def __init__(self):
        self.dnn_operator : DNNOperator = DNNOperator()
        self.config = None 
        self.hands = mp.solutions.hands.Hands()
        
    def process_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        dets = []
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                landmarks_y = [landmark.y for landmark in landmarks.landmark]   

                self.mp_drawing.draw_landmarks(image=rgb_frame,
                        landmark_list=landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.drawing_styles.get_default_hand_landmarks_style()
                )

                index_finger_y = int(landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * rgb_frame.shape[0])
                wrist_y = int(landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * rgb_frame.shape[0])
                index_finger_z = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].z
                wrist_z = landmarks.landmark[self.mp_hands.HandLandmark.WRIST].z
                
                if index_finger_y < wrist_y and index_finger_z < wrist_z - self.depth_threshold:
                    det = self.dnn_operator.dnn_classification(predict_input=landmarks_y,
                                                                output_size=6, 
                                                                output_map={0: 'BYE', 1:'GOOD', 2:'FIGTH', 3:'OK', 4: 'LOVE', 5: 'CALL'})
                    if det[1] >= 0.7:
                        dets.append(det[0])
                        cv2.putText(frame, f'{det[0]}: {det[1]:.2f}', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)                
                        for landmark in landmarks.landmark:
                            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        return frame, str() if (len(dets) == 0) else dets[0]