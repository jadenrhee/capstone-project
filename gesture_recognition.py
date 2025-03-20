import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the camera
camera_index = 1  # Change this index to the correct one for your camera
cap = cv2.VideoCapture(camera_index)

# Initialize variables to store landmarks
index_finger_tip = None
middle_finger_tip = None
thumb_tip = None
wrist = None
message = ""
is_pinching = False
is_swiping = False
is_tapping = False

# Initialize variables to store previous positions
prev_index_finger_tip = None
prev_middle_finger_tip = None

# Gesture detection thresholds
pinch_threshold = 20
swipe_velocity_threshold = 20
tap_distance_threshold = 50

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
                elif landmark == mp_hands.HandLandmark.WRIST:
                    wrist = (x, y)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Gesture recognition

    if index_finger_tip and thumb_tip:
        pinch_distance = ((index_finger_tip[0] - thumb_tip[0]) ** 2 + (index_finger_tip[1] - thumb_tip[1]) ** 2) ** 0.5
        if pinch_distance < pinch_threshold:
            message = "Pinch Detected"
        is_pinching = True
    else:
        is_pinching = False

    if middle_finger_tip and thumb_tip:
        pinch_distance = ((middle_finger_tip[0] - thumb_tip[0]) ** 2 + (middle_finger_tip[1] - thumb_tip[1]) ** 2) ** 0.5
        if pinch_distance < pinch_threshold:
            message = "Pinch Detected"
        is_pinching = True
    else:
        is_pinching = False

    if index_finger_tip and prev_index_finger_tip:
        index_finger_velocity = ((index_finger_tip[0] - prev_index_finger_tip[0]) ** 2 + (index_finger_tip[1] - prev_index_finger_tip[1]) ** 2) ** 0.5
        if index_finger_velocity > swipe_velocity_threshold:
            message = "Swipe Detected"
        is_swiping = True
    else:
        is_swiping = False

    if middle_finger_tip and prev_middle_finger_tip:
        middle_finger_velocity = ((middle_finger_tip[0] - prev_middle_finger_tip[0]) ** 2 + (middle_finger_tip[1] - prev_middle_finger_tip[1]) ** 2) ** 0.5
        if middle_finger_velocity > swipe_velocity_threshold:
            message = "Swipe Detected"
        is_swiping = True
    else:
        is_swiping = False

    # Check for no movement
    if prev_index_finger_tip and prev_middle_finger_tip:
        no_movement = ((index_finger_tip[0] - prev_index_finger_tip[0]) ** 2 + (index_finger_tip[1] - prev_index_finger_tip[1]) ** 2) ** 0.5 < 1 and \
                      ((middle_finger_tip[0] - prev_middle_finger_tip[0]) ** 2 + (middle_finger_tip[1] - prev_middle_finger_tip[1]) ** 2) ** 0.5 < 1
        if no_movement:
            message = ""

    if index_finger_tip and wrist:
        tap_distance = ((index_finger_tip[0] - wrist[0]) ** 2 + (index_finger_tip[1] - wrist[1]) ** 2) ** 0.5
        if tap_distance < tap_distance_threshold:
            message = "Tap Detected"
        is_tapping = True
    else:
        is_tapping = False

    if middle_finger_tip and wrist:
        tap_distance = ((middle_finger_tip[0] - wrist[0]) ** 2 + (middle_finger_tip[1] - wrist[1]) ** 2) ** 0.5
        if tap_distance < tap_distance_threshold:
            message = "Tap Detected"
        is_tapping = True
    else:
        is_tapping = False

    if message:
        cv2.putText(image, message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Update previous positions
    prev_index_finger_tip = index_finger_tip
    prev_middle_finger_tip = middle_finger_tip

    cv2.imshow('Hand Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('Hand Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
