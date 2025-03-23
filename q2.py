import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Game settings
width, height = 1280, 640
player_size = 100
player_start = [width // 2 - player_size // 2, height - player_size - 20]

# Enemy settings
enemy_speed = 5
enemy_size = 50
enemy_list = []

# Initialize score
score = 0

def create_enemy():
    x = random.randint(0, width - enemy_size)
    return [x, 0]

def move_enemies(enemy_list):
    global score
    updated_list = []
    for enemy in enemy_list:
        enemy[1] += enemy_speed
        if enemy[1] > height:
            score += 1
        else:
            updated_list.append(enemy)
    return updated_list

def check_collision(player_pos, enemy_list):
    for enemy in enemy_list:
        if (enemy[0] < player_pos[0] + player_size and
            enemy[0] + enemy_size > player_pos[0] and
            enemy[1] < player_pos[1] + player_size and
            enemy[1] + enemy_size > player_pos[1]):
            return True
    return False

def reset_game():
    global enemy_list, score, player_pos
    enemy_list = []
    score = 0
    player_pos = player_start.copy()

# Open cam
cap = cv2.VideoCapture(0)

player_pos = player_start.copy()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and resize frame
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Get coordinates of the index finger tip (landmark 8)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            x_coord = int(index_finger_tip.x * width)
            
            player_pos[0] = x_coord - player_size // 2

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    # Add new enemies
    if random.randint(1, 35) == 1:
        enemy_list.append(create_enemy())

    enemy_list = move_enemies(enemy_list)

    # Check for collision and enter game over mode
    if check_collision(player_pos, enemy_list):
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (width // 2 - 350, height // 2 - 150),
                      (width // 2 + 350, height // 2 + 150), (0, 0, 0), -1)
        cv2.putText(overlay, "GAME OVER", (width // 2 - 200, height // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(overlay, f"Score: {score}", (width // 2 - 150, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
        cv2.putText(overlay, "Press R to Restart", (width // 2 - 250, height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(overlay, "Press Q to Quit", (width // 2 - 230, height // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.imshow("Object Dodging Game", frame)
        
        # Restart & Quit options
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                reset_game()
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.rectangle(frame, (player_pos[0], player_pos[1]),
                  (player_pos[0] + player_size, player_pos[1] + player_size),
                  (0, 255, 0), -1)

    for enemy in enemy_list:
        cv2.rectangle(frame, (enemy[0], enemy[1]),
                      (enemy[0] + enemy_size, enemy[1] + enemy_size),
                      (0, 0, 255), -1)

    # Display score on frame
    cv2.rectangle(frame, (0, 0), (250, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"Score: {score}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.imshow("Object Dodging Game", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()