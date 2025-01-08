import cv2
import numpy as np
import eye
import os


def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        exit()
    return cap


def capture_images(cap):
    print("Press the SPACE bar to capture an image.")
    photo_counter = 0
    if not os.path.exists('calibrate'):
        os.mkdir('calibrate')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img = frame.copy()  # Create a copy of frame to modify

        y, x = frame.shape[:2]
        org = (int((x / 2) * 0.3), int(y / 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)  # White color in BGR
        thickness = 2
        radius = 10  # Circle radius

        if photo_counter == 0:
            # cv2.putText(img, 'Please look at the red dot on the top of the screen.', org,
            #             font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.circle(img, (int(x / 2), radius), radius, (0, 0, 255), -1)
        elif photo_counter == 1:
            # cv2.putText(img, 'Please look at the red dot on the bottom of the screen.', org,
            #             font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.circle(img, (int(x / 2), y - radius), radius, (0, 0, 255), -1)
        elif photo_counter == 2:
            # cv2.putText(img, 'Please look at the red dot on the left of the screen.', org,
            #             font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.circle(img, (radius, int(y / 2)), radius, (0, 0, 255), -1)
        elif photo_counter == 3:
            # cv2.putText(img, 'Please look at the red dot on the right of the screen.', org,
            #             font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.circle(img, (x - radius, int(y / 2)), radius, (0, 0, 255), -1)

        cv2.namedWindow('take_pic', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('take_pic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('take_pic', img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if photo_counter == 0:
                cv2.imwrite('calibrate/top.png', frame)
                print("Image captured and saved as 'top.png'")
            elif photo_counter == 1:
                cv2.imwrite('calibrate/bottom.png', frame)
                print("Image captured and saved as 'bottom.png'")
            elif photo_counter == 2:
                cv2.imwrite('calibrate/left.png', frame)
                print("Image captured and saved as 'left.png'")
            elif photo_counter == 3:
                cv2.imwrite('calibrate/right.png', frame)
                print("Image captured and saved as 'right.png'")

            photo_counter += 1

            if photo_counter >= 4:
                break
        elif key == ord('q'):
            break


def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()


def get_range():
    tracker = eye.EyeTracker()

    # [[left_eye_dist], [right_eye_dist]]: [[x, y], [x, y]]
    top = tracker.get_location(cv2.imread('calibrate/top.png'), display=False)
    bottom = tracker.get_location(cv2.imread('calibrate/bottom.png'), display=False)
    left = tracker.get_location(cv2.imread('calibrate/left.png'), display=False)
    right = tracker.get_location(cv2.imread('calibrate/right.png'), display=False)

    left_eye_x_range = [left[0][0], right[0][0]]
    left_eye_y_range = [top[0][1], bottom[0][1]]
    right_eye_x_range = [left[1][0], right[1][0]]
    right_eye_y_range = [top[1][1], bottom[1][1]]

    return left_eye_x_range, right_eye_x_range, left_eye_y_range, right_eye_y_range


def run():
    cap = initialize_camera()
    capture_images(cap)
    release_camera(cap)
    left_eye_x_range, right_eye_x_range, left_eye_y_range, right_eye_y_range = get_range()

    # left_eye_x_range: [left, right]
    # right_eye_x_range: [left, right]
    # left_eye_y_range: [top, bottom]
    # right_eye_y_range: [top, bottom]

    return left_eye_x_range, right_eye_x_range, left_eye_y_range, right_eye_y_range



if __name__ == "__main__":
    run()
