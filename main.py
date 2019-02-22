import sys
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/vlados/Jupyter/Road_project')

from config import Config
from birdeye_view import Birdeye_view
from Drawing_lane import Drawing_lane

bev = Birdeye_view()
dl = Drawing_lane()

cap = cv2.VideoCapture(Config.video_path) 
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    start_time = time.time()
    ret, frame = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        break
    
    dst = bev.get_perspective_warp(frame)
    frame = dl.find_lane(dst, frame)

    cv2.putText(frame, f'FPS = {1/(time.time() - start_time):.3f}', (10,50), font, 1,(255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()

