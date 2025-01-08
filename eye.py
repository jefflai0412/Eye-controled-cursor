import cv2
import numpy as np
import dlib
import time
from skimage.exposure import match_histograms


def calculate_fps(prev_time):
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    return fps, prev_time


class EyeTracker:
    def __init__(self):
        self.frame_original = None
        self.frame_display = None

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.face_area = None
        self.face_frame = None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

        self.eye_y_top = None
        self.eye_y_bottom = None
        self.eye_x_right = None
        self.eye_x_left = None

        self.left_eye_area = None
        self.left_eye_frame = None
        self.right_eye_area = None
        self.right_eye_frame = None

        self.left_eye_keypoint = None
        self.right_eye_keypoint = None

        self.right_eye = None
        self.left_eye = None

        self.eyes_location = []

        self.left_eye_dist = []
        self.right_eye_dist = []

        self.left_eye_closed = False
        self.close_start_time = None

        self.left_click = False

    def pre_process(self, frame):
        img = cv2.flip(frame, 1)
        self.frame_display = img
        self.frame_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    def face_detect(self):
        faces = self.face_cascade.detectMultiScale(self.frame_original, 1.3, 2)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)  # Sort by area
        if faces:
            fx, fy, fw, fh = faces[0]
            self.face_area = fx, fy, fw, fh
            self.face_frame = self.frame_original[fy:fy + round(fh), fx:fx + fw]
            # Display
            cv2.rectangle(self.frame_display, (fx, fy), (fx + fw, fy + round(fh)), (255, 0, 0), 2)

    def face_keypoint(self):
        if self.face_frame is None:
            return
        faces = self.detector(self.face_frame, 1)
        for face in faces:
            landmarks = self.predictor(self.face_frame, face)

            left_eye_top = min(landmarks.part(36).y, landmarks.part(37).y, landmarks.part(38).y,
                               landmarks.part(39).y) - 10
            left_eye_bottom = max(landmarks.part(36).y, landmarks.part(41).y, landmarks.part(40).y,
                                  landmarks.part(39).y) + 10
            left_eye_left = landmarks.part(36).x - 10
            left_eye_right = landmarks.part(39).x + 10

            right_eye_top = min(landmarks.part(42).y, landmarks.part(43).y, landmarks.part(44).y,
                                landmarks.part(45).y) - 10
            right_eye_bottom = max(landmarks.part(42).y, landmarks.part(47).y, landmarks.part(46).y,
                                   landmarks.part(45).y) + 10
            right_eye_left = landmarks.part(42).x - 10
            right_eye_right = landmarks.part(45).x + 10

            # for n in range(36, 48):
            #     x = landmarks.part(n).x
            #     y = landmarks.part(n).y
            #     cv2.circle(self.frame_display, (x + self.face_area[0], y + self.face_area[1]), 2, (0, 255, 0), -1)

            # calculate the closed duration
            self.left_click = False
            if landmarks.part(41).y - landmarks.part(37).y < 7 and self.left_eye_closed == False:
                self.left_eye_closed = True
                self.close_start_time = time.time()
                self.left_eye_frame = None
                return
            elif landmarks.part(41).y - landmarks.part(37).y >= 7 and self.left_eye_closed == True:
                self.left_eye_closed = False
                if time.time() - self.close_start_time > 2:
                    self.left_click = True
                    print("left click")

            self.left_eye_area = left_eye_left, left_eye_top, left_eye_right - left_eye_left, left_eye_bottom - left_eye_top  # x, y, w, h:based on face frame
            self.right_eye_area = right_eye_left, right_eye_top, right_eye_right - right_eye_left, right_eye_bottom - right_eye_top

            self.left_eye_keypoint = [landmarks.part(39).x, landmarks.part(39).y]  # based on face frame
            self.right_eye_keypoint = [landmarks.part(42).x, landmarks.part(42).y]  # based on face frame

            self.left_eye_frame = self.face_frame[self.left_eye_area[1]:self.left_eye_area[1] + self.left_eye_area[3],
                                  self.left_eye_area[0]:self.left_eye_area[0] + self.left_eye_area[2]]
            self.right_eye_frame = self.face_frame[
                                   self.right_eye_area[1]:self.right_eye_area[1] + self.right_eye_area[3],
                                   self.right_eye_area[0]:self.right_eye_area[0] + self.right_eye_area[2]]

            # Display
            cv2.putText(self.frame_display, 'left eye ROI',
                        (self.face_area[0] + self.left_eye_area[0], self.face_area[1] + self.left_eye_area[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(self.frame_display,
                          (self.face_area[0] + self.left_eye_area[0], self.face_area[1] + self.left_eye_area[1]), (
                              self.face_area[0] + self.left_eye_area[0] + self.left_eye_area[2],
                              self.face_area[1] + self.left_eye_area[1] + self.left_eye_area[3]), (0, 255, 0), 2)

            cv2.putText(self.frame_display, 'right eye ROI',
                        (self.face_area[0] + self.right_eye_area[0], self.face_area[1] + self.right_eye_area[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(self.frame_display,
                          (self.face_area[0] + self.right_eye_area[0], self.face_area[1] + self.right_eye_area[1]), (
                              self.face_area[0] + self.right_eye_area[0] + self.right_eye_area[2],
                              self.face_area[1] + self.right_eye_area[1] + self.right_eye_area[3]), (0, 255, 0), 2)

    def pupil_detect(self):
        if self.left_eye_frame is None or self.right_eye_frame is None:
            return

        self.eyes_location = []  # Clear eye location list

        left_img = self.left_eye_frame.copy()
        right_img = self.right_eye_frame.copy()
        cv2.imshow('left eye', left_img)
        cv2.moveWindow('left eye', 0, 0)

        reference = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)
        left_match = match_histograms(left_img, reference)
        right_match = match_histograms(right_img, reference)
        cv2.imshow('left eye hist', left_match)

        left_filtered = cv2.bilateralFilter(left_img, 9, 75, 75)
        right_filtered = cv2.bilateralFilter(right_img, 9, 75, 75)
        cv2.imshow('left eye filtered', left_filtered)
        cv2.moveWindow('left eye filtered', 0, 100)

        # left_threshold = cv2.adaptiveThreshold(left_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # right_threshold = cv2.adaptiveThreshold(right_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        left_threshold = cv2.threshold(left_filtered, 70, 255, cv2.THRESH_BINARY)[1]
        right_threshold = cv2.threshold(right_filtered, 70, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('left eye threshold', left_threshold)
        cv2.moveWindow('left eye threshold', 0, 200)

        left_reverse = cv2.bitwise_not(left_threshold)
        right_reverse = cv2.bitwise_not(right_threshold)
        cv2.imshow('left eye reverse', left_reverse)
        cv2.moveWindow('left eye reverse', 0, 300)

        kernel = np.ones((8, 5), np.uint8)
        left_opened = cv2.morphologyEx(left_reverse, cv2.MORPH_OPEN, kernel, iterations=1)
        right_opened = cv2.morphologyEx(right_reverse, cv2.MORPH_OPEN, kernel, iterations=1)
        cv2.imshow('left eye opened', left_opened)
        cv2.moveWindow('left eye opened', 0, 400)

        # left_erosion = cv2.erode(left_opened, kernel, iterations=1)
        # right_erosion = cv2.erode(right_opened, kernel, iterations=1)
        # cv2.imshow('left eye erosion', left_erosion)
        # cv2.imshow('right eye erosion', right_erosion)

        left_contours, _ = cv2.findContours(left_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        left_contours = sorted(left_contours, key=lambda x: cv2.contourArea(x), reverse=True)

        right_contours, _ = cv2.findContours(right_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        right_contours = sorted(right_contours, key=lambda x: cv2.contourArea(x), reverse=True)

        if left_contours and right_contours:  # Ensure there are contours found
            left_contour = left_contours[0]
            right_contour = right_contours[0]

            left_moments = cv2.moments(left_contour)
            right_moments = cv2.moments(right_contour)

            if left_moments['m00'] != 0 and right_moments['m00'] != 0:  # Avoid division by zero
                left_cx = int(left_moments['m10'] / left_moments['m00'])
                left_cy = int(left_moments['m01'] / left_moments['m00'])

                right_cx = int(right_moments['m10'] / right_moments['m00'])
                right_cy = int(right_moments['m01'] / right_moments['m00'])

                cv2.circle(self.frame_display, (left_cx + self.left_eye_area[0] + self.face_area[0],
                                                left_cy + self.left_eye_area[1] + self.face_area[1]), 2, (0, 0, 255), 7)
                cv2.circle(self.frame_display, (right_cx + self.right_eye_area[0] + self.face_area[0],
                                                right_cy + self.right_eye_area[1] + self.face_area[1]), 2, (0, 0, 255),
                           7)

                self.eyes_location = [(left_cx + self.left_eye_area[0], left_cy + self.left_eye_area[1]),
                                      (right_cx + self.right_eye_area[0], right_cy + self.right_eye_area[1])]

    def eye_distance(self):  # Calculate the distance between the pupil and the reference point of the eye socket
        if len(self.eyes_location) < 2:
            return None
        self.left_eye_dist = [self.eyes_location[0][0] - self.left_eye_keypoint[0],
                              self.eyes_location[0][1] - self.left_eye_keypoint[1]]
        self.right_eye_dist = [self.eyes_location[1][0] - self.right_eye_keypoint[0],
                               self.eyes_location[1][1] - self.right_eye_keypoint[1]]

    # External call function
    def get_location(self, frame, display=False):
        self.pre_process(frame)
        self.face_detect()
        self.face_keypoint()
        self.pupil_detect()
        self.eye_distance()
        # print(self.left_eye_dist, self.right_eye_dist)

        if display:
            cv2.imshow('frame', self.frame_display)
            cv2.waitKey(0)

        return self.left_eye_dist, self.right_eye_dist  # [x, y], [x, y]


# Internal execution function
def main():
    tracker = EyeTracker()
    prev_time = 0
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 450, 300)

    while True:
        ret, frame = tracker.cap.read()
        if not ret:
            break
        left_eye, right_eye = tracker.get_location(frame, display=False)
        # [x, y], [x, y]

        # Calculate fps
        fps, prev_time = calculate_fps(prev_time)

        # Display FPS on frame
        cv2.putText(tracker.frame_display, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Set window to be always on top
        # cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('frame', tracker.frame_display)

        # print frame size
        print(tracker.frame_display.shape)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
