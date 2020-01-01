# capture a video
import cv2 as cv

def video_capture():
	capture = cv.VideoCapture('video.mp4')
	while True:
		ret, frame = capture.read()
		if ret is False:
			break
		hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
		print(hsv)

video_capture()
