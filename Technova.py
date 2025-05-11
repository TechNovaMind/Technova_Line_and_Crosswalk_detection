# Technova line follower + crosswalk spotter
# Jetson Nano with OAK-D, talks to Raspberry Pi 4
# Made it look like I wrote it in a rush, please don’t crash...:)

import cv2
import depthai as dai
import numpy as np
import socket
import msgpack
import time

# Display flags
SHOW_DISPLAY = True  # Flip to False for speed
SHOW_WARP = True  # False to skip warp view
IMG_SIZE = (400, 400)  # OG size

# Crosswalk Detection setting(change to find the best result according to your project..)
Min_bands = 7  #depends on your Crosswalk's shape and size
Max_bands = 14 #----
space_mm = 18
width_mm = 12
PIXEL_CONVERT = 4.2 # change to find the best result
Gap_pix = space_mm * PIXEL_CONVERT
Width_pix = width_mm * PIXEL_CONVERT
Min_area = 55 #Change base on your project
SHAPE_RATIO = 1.5
Frame_num_confirmation = 5

# Crosswalk setting
bands_count = 0
crosswalk_sure = False
in_crosswalk = False
stop_time = 0

# Pi connection
PI_IP = "192.168.2.2"
PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.01)

# PID setting..
kp_steering = 0.3
ki_steering = 0.05
kd_steering = 0.08
steer_integral = 0
last_steer_err = 0
integral_max = 20
kp_speed = 0.5
kd_speed = 0.3
last_speed_err = 0

servo = 90
Servo_min_angle = 25
Servo_max_angle = 155
Technova_Speed = 30

# Camera setup
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.MonoCamera)
out = pipeline.create(dai.node.XLinkOut)
out.setStreamName("right")
cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam.setCamera("right")
cam.out.link(out.input)

# Perspective points...
src_points = np.float32([
    [100, 320],  # top-left
    [350, 320],  # top-right
    [50, 400],   # bottom-left
    [400, 400]   # bottom-right
])
dst_points = np.float32([
    [0, 0],
    [400, 0],
    [0, 400],
    [400, 400]
])
warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
inv_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

# Sliders Setting...
cv2.namedWindow("Sliders")
cv2.createTrackbar("Canny Min", "Sliders", 130, 255, lambda x: None)
cv2.createTrackbar("Canny Max", "Sliders", 150, 255, lambda x: None)
cv2.createTrackbar("Hough Thresh", "Sliders", 30, 200, lambda x: None)
cv2.createTrackbar("Min Line", "Sliders", 50, 200, lambda x: None)
cv2.createTrackbar("Max Gap", "Sliders", 60, 100, lambda x: None)

# Check Bands...
def check_bands(boxes):
    if not (Min_bands <= len(boxes) <= Max_bands):
        print(f"Stripes wrong: {len(boxes)}, need 7-14")
        return False
    boxes = sorted(boxes, key=lambda b: b[0])
    for _, _, w, _ in boxes:
        if not (Width_pix * 0.1 < w < Width_pix * 4):
            print(f"Width {w} is odd, want ~{Width_pix}")
            return False
    gaps = [boxes[i+1][0] - boxes[i][0] for i in range(len(boxes) - 1)]
    for g in gaps:
        if not (Gap_pix * 0.1 < g < Gap_pix * 4):
            print(f"Gap {g} off, want ~{Gap_pix}")
            return False
    return True

# Crosswalk detection
def find_crosswalk(frame, disp=None, warp=None):
    global bands_count, crosswalk_sure, in_crosswalk
    print("Warping...")
    warped = cv2.warpPerspective(frame, warp_matrix, (400, 120))
    if SHOW_WARP and warp is not None:
        warp[:] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    gray = cv2.GaussianBlur(warped, (3, 3), 0)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 100, 200)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ratio = max(w, h) / max(1, min(w, h))
        if area > Min_area and ratio > SHAPE_RATIO:
            pts = np.float32([[[x, y]], [[x + w, y + h]]])
            mapped = cv2.perspectiveTransform(pts, inv_matrix)
            x1, y1 = mapped[0][0]
            x2, y2 = mapped[1][0]
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            stripe_boxes.append((x, y, w, h))
            if SHOW_DISPLAY and disp is not None:
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if SHOW_WARP and warp is not None:
                cv2.rectangle(warp, (x, y), (x + w, y + h), (0, 255, 0), 1)

    print(f"Stripes: {len(stripe_boxes)}")
    max_y = max([b[1] + b[3] for b in stripe_boxes]) if stripe_boxes else 0
    frame_h = frame.shape[0]

    if check_bands(stripe_boxes):
        if not in_crosswalk and max_y > frame_h * 0.8:
            bands_count += 1
            if bands_count >= Frame_num_confirmation:
                crosswalk_sure = True
                in_crosswalk = True
                print("Yesss, crosswalk :)")
            return True, stripe_boxes
    else:
        if in_crosswalk and len(stripe_boxes) < 5:
            in_crosswalk = False
            bands_count = 0
            crosswalk_sure = False
            print("Crosswalk done...")
        return False, []
    return False, stripe_boxes

# Line detection Setting
def process_lines(frame):
    t = time.time()
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    low = cv2.getTrackbarPos("Canny Min", "Sliders")
    high = cv2.getTrackbarPos("Canny Max", "Sliders")
    edges = cv2.Canny(blur, low, high)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    roi = np.array([[(0, h), (w, h), (w//2 + 250, h//2), (w//2 - 250, h//2)]])
    cv2.fillPoly(mask, roi, 255)
    masked = cv2.bitwise_and(edges, mask)
    thresh = cv2.getTrackbarPos("Hough Thresh", "Sliders")
    min_len = cv2.getTrackbarPos("Min Line", "Sliders")
    max_gap = cv2.getTrackbarPos("Max Gap", "Sliders")
    lines = cv2.HoughLinesP(masked, 1, np.pi/180, thresh, minLineLength=min_len, maxLineGap=max_gap)
    if lines is not None:
        left_x = []
        right_x = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < w//2 and x2 < w//2:
                left_x.extend([x1, x2])
            else:
                right_x.extend([x1, x2])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        left = int(np.mean(left_x)) if left_x else w//4
        right = int(np.mean(right_x)) if right_x else 3*w//4
        print(f"Lines: {left}, {right}")
        return left, right
    print("No lines detect :(")
    return None, None

# Main loop ...
with dai.Device(pipeline) as device:
    queue = device.getOutputQueue("right", maxSize=4, blocking=False)
    print("Technova's going...")
    cv2.startWindowThread()

    while True:
        try:
            t = time.time()
            frame = queue.get().getCvFrame()

            # Crosswalk
            disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if SHOW_DISPLAY else None
            warp = np.zeros((120, 400, 3), np.uint8) if SHOW_WARP else None
            detected, stripes = find_crosswalk(frame, disp, warp)

            # Lines
            left, right = process_lines(frame)
            image_center = frame.shape[1] // 2

            # Control technova....
            move_cmd = "FORWARD"
            if crosswalk_sure and (time.time() - stop_time) < 5:
                move_cmd = "STOP"
                Technova_Speed = 0
                servo = 90
                print("Technova: Stopped")
            elif crosswalk_sure and (time.time() - stop_time) < 10:
                move_cmd = "FORWARD"
                Technova_Speed = 30
                servo = 90
                print("Technova: Crossing")
                if time.time() - stop_time >= 10:
                    crosswalk_sure = False
            elif detected:
                move_cmd = "STOP"
                Technova_Speed = 0
                servo = 90
                print("Technova: Crosswalk!")
                stop_time = time.time()
                crosswalk_sure = True
            elif in_crosswalk:
                move_cmd = "FORWARD"
                Technova_Speed = 30
                servo = 90
                print("Technova: In crosswalk")
            else:
                Technova_Speed = 30
                if left is not None and right is not None:
                    lane_center = (left + right) // 2
                    err = image_center - lane_center
                    d_err = err - last_steer_err
                    steer_integral += err * 0.01
                    steer_integral = np.clip(steer_integral, -integral_max, integral_max)
                    last_steer_err = err
                    steer = kp_steering * err + ki_steering * steer_integral + kd_steering * d_err
                    servo = np.clip(90 - steer + 10, Servo_min_angle, Servo_max_angle)
                    speed_err = abs(err)
                    d_speed_err = speed_err - last_speed_err
                    last_speed_err = speed_err
                    speed_adj = kp_speed * speed_err + kd_speed * d_speed_err
                    Technova_Speed = np.clip(50 - speed_adj, 30, 70)
                    print(f"Steering: {servo}, Speed: {Technova_Speed}")
                else:
                    move_cmd = "STOP"
                    Technova_Speed = 0
                    servo = 90

            # Send to Raspberry ..
            try:
                sock.sendto(msgpack.packb({"command": "STEER", "angle": int(servo)}), (PI_IP, PORT))
                sock.sendto(msgpack.packb({"command": "MOVE", "direction": move_cmd, "speed": int(Technova_Speed)}), (PI_IP, PORT))
            except socket.error:
                print("Socket’s down, fixing...")
                sock.close()
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(0.01)

            # Display Stufff...
            if SHOW_DISPLAY and disp is not None:
                text = "Crosswalk :)" if detected else "In Crosswalk.." if in_crosswalk else "NOOO"
                color = (0, 0, 255) if detected else (0, 255, 255) if in_crosswalk else (255, 0, 0)
                cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                fps = 1 / (time.time() - t)
                cv2.putText(disp, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imshow("Crosswalk", disp)
            if SHOW_WARP and warp is not None:
                cv2.imshow("Perspective", warp)
            #cv2.imshow("Cam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Noooo, loop crashed : {e}")
            time.sleep(0.01)

# Cleanup
try:
    sock.sendto(msgpack.packb({"command": "MOVE", "direction": "STOP", "speed": 0}), (PI_IP, PORT))
except socket.error:
    print("Cleanup messed up")
sock.close()
cv2.destroyAllWindows()