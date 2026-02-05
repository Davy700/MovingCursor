import cv2
import mediapipe as mp
import pyautogui
import mouse
import time

screen = pyautogui.size()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

counter = 0

mouse_x, mouse_y = screen.width/2, screen.height/2

mouse.move(mouse_x, mouse_y)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            thumb_tip= hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            minus_thumb_pos = 1000*(thumb_mcp.y - thumb_tip.y)

            if (minus_thumb_pos < 30 and minus_thumb_pos > -30) and counter == 0:
                mouse.click('left')
                counter = 40

            mouse_x = screen.width/100*((index_finger_tip.x)*100)
            mouse_y = screen.height/100*(index_finger_tip.y*100)

            mouse.move(mouse_x, mouse_y, True)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Hand Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if counter > 0:
            counter -= 1


cap.release()
cv2.destroyAllWindows()
