import cv2
import mediapipe as mp
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the camera
camera_index = 1 # Change this index to the correct one for your camera
cap = cv2.VideoCapture(camera_index)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Mirror the image
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    hand_results = hands.process(image_rgb)

    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # Recognize predefined gestures
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Assign variables to all the joints and tips for both hands
            for landmark in mp_hands.HandLandmark:
                x = int(hand_landmarks.landmark[landmark].x * image.shape[1])
                y = int(hand_landmarks.landmark[landmark].y * image.shape[0])
                if landmark == mp_hands.HandLandmark.THUMB_TIP:
                    thumb_tip = (x, y)
                elif landmark == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                    index_finger_tip = (x, y)
                elif landmark == mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
                    middle_finger_tip = (x, y)
                elif landmark == mp_hands.HandLandmark.RING_FINGER_TIP:
                    ring_finger_tip = (x, y)
                elif landmark == mp_hands.HandLandmark.PINKY_TIP:
                    pinky_tip = (x, y)
                elif landmark == mp_hands.HandLandmark.WRIST:
                    wrist = (x, y)
                # Add more landmarks as needed
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Gesture recognition
    if 'index_finger_tip' in locals() and 'thumb_tip' in locals():
        pinch_distance = ((index_finger_tip[0] - thumb_tip[0]) ** 2 + (index_finger_tip[1] - thumb_tip[1]) ** 2) ** 0.5
        if pinch_distance < 40:
            cv2.putText(image, "Pinch Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if 'index_finger_tip' in locals() and 'middle_finger_tip' in locals():
        swipe_distance = ((index_finger_tip[0] - middle_finger_tip[0]) ** 2 + (index_finger_tip[1] - middle_finger_tip[1]) ** 2) ** 0.5
        if swipe_distance > 100:
            cv2.putText(image, "Swipe Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if 'index_finger_tip' in locals() and 'wrist' in locals():
        tap_distance = ((index_finger_tip[0] - wrist[0]) ** 2 + (index_finger_tip[1] - wrist[1]) ** 2) ** 0.5
        if tap_distance < 50:
            cv2.putText(image, "Tap Detected", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('Hand Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
