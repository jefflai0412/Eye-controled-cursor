import time
import threading
import tkinter as tk
from collections import deque
from random import random

import calibrate
import eye
import cv2
import pyautogui as pag
import sys

pag.FAILSAFE = False

# root1 = tk.Tk()
# label2 = tk.Label(root1, text='0', width=50)

mouseX_positions = deque(maxlen=5)
mouseY_positions = deque(maxlen=5)

def calPosition(eye_area, eye_location):
    global mouse_position
    # Get screen size
    screenWidth, screenHeight = pag.size()
    # Calculate mouse position
    if len(eye_location) < 2:
        mouse_position = (screenWidth / 2, screenHeight / 2)
    else:
        mouseX = (eye_location[0] - eye_area[0][0]) * screenWidth / (eye_area[0][1] - eye_area[0][0])
        mouseY = (eye_location[1] - eye_area[1][0]) * screenHeight / (eye_area[1][1] - eye_area[1][0])
        mouse_position = (mouseX, mouseY)

    # label2['text'] = f'瞳孔座標： ({eye_location[0]}, {eye_location[1]}), 滑鼠座標：({mouseX}, {mouseY})'
    sys.stdout.flush()
    time.sleep(0.005)
    return mouse_position


def control_mouse(eye_area, eye_coordinate):
    calculated_position = calPosition(eye_area, eye_coordinate)
    mouseX_positions.append(calculated_position[0])
    mouseY_positions.append(calculated_position[1])
    target_position = (sum(mouseX_positions) / len(mouseX_positions), sum(mouseY_positions) / len(mouseY_positions))
    pag.moveTo(target_position[0], target_position[1], duration=0.005)


def main():
    left_eye_x_range, right_eye_x_range, left_eye_y_range, right_eye_y_range = calibrate.run()
    # [left, right], [left, right], [top, bottom], [top, bottom]
    print("left_eye_x_range: ", left_eye_x_range)
    print("left_eye_y_range: ", left_eye_y_range)

    left_eye_range = [left_eye_x_range, left_eye_y_range]  # [left, right], [top, bottom]
    right_eye_range = [right_eye_x_range, right_eye_y_range]  # [left, right], [top, bottom]

    cap = cv2.VideoCapture(0)

    tracker = eye.EyeTracker()  # changed
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 450, 300)

    while True:
        ret, frame = cap.read()
        if ret:
            left_eye_coordinate, right_eye_coordinate = tracker.get_location(frame, display=False)  # changed # [x, y], [x, y]
            # print("left_eye_coordinate: ", left_eye_coordinate)
            # print("right_eye_coordinate: ", right_eye_coordinate)
            control_mouse(left_eye_range, left_eye_coordinate)  # changed
            if tracker.left_click:
                pag.click(button='left')
                print("left click")


            # Set window to be always on top
            cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)

            cv2.imshow('frame', tracker.frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Can't receive frame.")
            break



if __name__ == "__main__":
    main()
