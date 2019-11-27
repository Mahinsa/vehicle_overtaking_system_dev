import cv2
import numpy as np
from matplotlib import pyplot as plt
from imports.vehicleDetection import BoundingBox
from imports.laneDetection import laneLines
import pyttsx3
import engineio

#initialise the speech engine
engineio = pyttsx3.init()
voices = engineio.getProperty('voices')
engineio.setProperty('rate', 130)    # Aquí puedes seleccionar la velocidad de la voz
engineio.setProperty('voice',voices[0].id)

capture = cv2.VideoCapture('video.mp4')
bb = BoundingBox()
ll = laneLines()
writer = None
# timeBetweenFrames = 2.04 

while True:
    ret, frame = capture.read()
    if ret == False:
        print("no frames")
        break
#     resized = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
#     final_image = ll.lane_line_process(frame)
    Det_image = bb.getBoundingBox(frame, engineio)
    if Det_image is None:
        continue
    cv2.imshow("video", Det_image)
    if cv2.waitKey(1) & 0xFF == ord("a"):
        break
#    if writer is None:
#    # initialize our video writer
#        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#        writer = cv2.VideoWriter("output//output004.avi", fourcc, 30,
#                             (frame.shape[1], frame.shape[0]), True)
#
#    # write the output frame to disk
#    writer.write(Det_image)
#
## release the file pointers
#print("[INFO] cleaning up...")
#writer.release()
 
capture.release()
cv2.destroyAllWindows()