import cv2
import numpy as np
import matplotlib.pyplot as plt

from birdeye_view import Birdeye_view

class Data_interpritation:        
 
    def __init__(self):
        self.frames = []
    
    def video_vizualizing(self, video_path):    
        """
            Opening the video and playing it

            video_path - path to the video
        """
        cap = cv2.VideoCapture(video_path)
        
        while(True):
            ret, frame = cap.read()
            
            if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
            cv2.imshow('Video', frame)

        cap.release()
        cv2.destroyAllWindows()
        
    def frame2img(self, video_path, save_per_frames=100):
        """
            Video playing and frames capchuring with interval 
            indicated in save_per_frames variable.

            video_path - path to the video
            save_per_vrame - saving interval. Each 100-th frame 
                             will be appended in self.frames dictionary
        """
        cap = cv2.VideoCapture(video_path) 

        frame_counter = 0

        while(True):
            ret, frame = cap.read() 

            if frame_counter % save_per_frames == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)

            frame_counter += 1

            if ret == False:
                break

        cap.release()
        cv2.destroyAllWindows()

        return np.array(self.frames)
    
    def img_vizualizing(self, frames, columns=5):
        """
            Subploting images. Images must be located in one dictionary.

            frames - dictionary with images
            columns - columns in in subplot grid
        """
        row = len(self.frames)//columns

        fig, ax = plt.subplots(row, columns, figsize=(21, 10), sharex='col', sharey='row')

        for i, axi in enumerate(ax.flat):
            axi.imshow(frames[i], cmap='gray')
