import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create the Hands object for hand detection
hands = mp_hands.Hands()

# Open the camera
cap = cv2.VideoCapture(1)  # Use 0 instead of 1 for default camera

# List to store hand landmarks
landmarks_data = []

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            landmarks_y = [landmark.y for landmark in landmarks.landmark]
            landmarks_x = [landmark.x for landmark in landmarks.landmark]

            for idx, y in enumerate(landmarks_y):
                cv2.putText(image, f'{idx}: {y:.2f}',
                            (int(landmarks.landmark[idx].x * image.shape[1]), int(y * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            landmarks_data.append(landmarks_y + landmarks_x)
            print(landmarks_data)

        elif key == ord('s'):
            df = pd.DataFrame(landmarks_data, columns=[f'Landmark_{i}' for i in range(len(landmarks_data[0]))])
            df.to_csv('byebye.csv', index=False)
            print('Save successful')
            landmarks_data = []  # Clear the landmarks data after saving

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
