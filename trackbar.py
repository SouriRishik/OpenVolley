import cv2
import numpy as np

lower_yellow_default = np.array([10, 80, 80])
upper_yellow_default = np.array([38, 225, 255])

lower_red_default = np.array([173, 152, 149])
upper_red_default = np.array([179, 255, 255])

cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)

def nothing(x):
    pass

cv2.createTrackbar('Lower H', 'Trackbars', lower_yellow_default[0], 179, nothing)
cv2.createTrackbar('Lower S', 'Trackbars', lower_yellow_default[1], 255, nothing)
cv2.createTrackbar('Lower V', 'Trackbars', lower_yellow_default[2], 255, nothing)
cv2.createTrackbar('Upper H', 'Trackbars', upper_yellow_default[0], 179, nothing)
cv2.createTrackbar('Upper S', 'Trackbars', upper_yellow_default[1], 255, nothing)
cv2.createTrackbar('Upper V', 'Trackbars', upper_yellow_default[2], 255, nothing)

cv2.createTrackbar('Lower H (Russia)', 'Trackbars', lower_red_default[0], 179, nothing)
cv2.createTrackbar('Lower S (Russia)', 'Trackbars', lower_red_default[1], 255, nothing)
cv2.createTrackbar('Lower V (Russia)', 'Trackbars', lower_red_default[2], 255, nothing)
cv2.createTrackbar('Upper H (Russia)', 'Trackbars', upper_red_default[0], 179, nothing)
cv2.createTrackbar('Upper S (Russia)', 'Trackbars', upper_red_default[1], 255, nothing)
cv2.createTrackbar('Upper V (Russia)', 'Trackbars', upper_red_default[2], 255, nothing)

def player_detection_brazil(frame, lower_yellow, upper_yellow):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 350:
            player_boxes.append((x, y, w, h))

    return player_boxes

def player_detection_russia(frame, lower_red, upper_red):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 500:
            player_boxes.append((x, y, w, h))

    return player_boxes

def detect_ball(frame, lower_ball_color, upper_ball_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_ball_color, upper_ball_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_position = None
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            x, y, w, h = cv2.boundingRect(contour)
            ball_position = (int(x + w / 2), int(y + h / 2))
            break

    return ball_position

image_path = "volleyball_image.jpg"
frame = cv2.imread(image_path)

while True:
    frame_resized = cv2.resize(frame, (640, 480))

    lower_yellow = np.array([cv2.getTrackbarPos('Lower H', 'Trackbars'),
                             cv2.getTrackbarPos('Lower S', 'Trackbars'),
                             cv2.getTrackbarPos('Lower V', 'Trackbars')])
    upper_yellow = np.array([cv2.getTrackbarPos('Upper H', 'Trackbars'),
                             cv2.getTrackbarPos('Upper S', 'Trackbars'),
                             cv2.getTrackbarPos('Upper V', 'Trackbars')])
    
    lower_red = np.array([cv2.getTrackbarPos('Lower H (Russia)', 'Trackbars'),
                          cv2.getTrackbarPos('Lower S (Russia)', 'Trackbars'),
                          cv2.getTrackbarPos('Lower V (Russia)', 'Trackbars')])
    upper_red = np.array([cv2.getTrackbarPos('Upper H (Russia)', 'Trackbars'),
                          cv2.getTrackbarPos('Upper S (Russia)', 'Trackbars'),
                          cv2.getTrackbarPos('Upper V (Russia)', 'Trackbars')])

    ball_position = detect_ball(frame_resized, lower_yellow, upper_yellow)

    players_brazil = player_detection_brazil(frame_resized, lower_yellow, upper_yellow)

    players_russia = player_detection_russia(frame_resized, lower_red, upper_red)

    frame_resized_with_boxes = frame_resized.copy()
    
    for player_bbox in players_brazil:
        x, y, w, h = player_bbox
        cv2.rectangle(frame_resized_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 2) 

    for player_bbox in players_russia:
        x, y, w, h = player_bbox
        cv2.rectangle(frame_resized_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if ball_position:
        cv2.circle(frame_resized_with_boxes, ball_position, 5, (0, 255, 255), -1)

    cv2.imshow("Volleyball Match", frame_resized_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
