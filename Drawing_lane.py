import sys
import cv2
import numpy as np

from birdeye_view import Birdeye_view

class Drawing_lane(Birdeye_view):

    def get_hist(self, img):
        return np.sum(img[len(img)//2:,:], axis=0)

    def hist_max(self, histogram):
        # find peaks of left and right halves
        midpoint = np.int(len(histogram)//2.5)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        return leftx_base, rightx_base

    def sliding_window(self, img):    
        out_img = np.dstack((img, img, img))*255
        histogram = self.get_hist(img)

        # Current positions to be updated for each window
        leftx_base, rightx_base = self.hist_max(histogram)
        # Set height of windows
        window_height = img.shape[0] // self.nwindows
        # Identify the x and y positions of all nonzero pixels in the image
        nonzeroy, nonzerox = img.nonzero()

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_base - self.margin
            win_xleft_high = leftx_base + self.margin
            win_xright_low = rightx_base - self.margin
            win_xright_high = rightx_base + self.margin
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
     
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_base = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_base = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
    #     print(leftx.shape, lefty.shape, rightx.shape, righty.shape)
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
        
        return out_img, (left_fitx, right_fitx), ploty


    def draw_lanes(self, frame, left_fit, right_fit):
        img = frame[self.left_up_corner_point:(self.left_up_corner_point + self.IMG_H), 0:self.IMG_W]
        ploty = np.linspace(0, (self.IMG_H - 1), self.IMG_H)
        color_img = np.zeros_like(img)
        
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))

        cv2.fillPoly(color_img, np.int_(points), (0,0,255))

        inv_perspective = self.inv_perspective_warp(color_img)
        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)

        frame[self.left_up_corner_point:(self.left_up_corner_point + self.IMG_H), 0:self.IMG_W] = inv_perspective
        
        return frame

    def find_lane(self, img, frame):
        out_img, curves, ploty = self.sliding_window(img)
        img_ = self.draw_lanes(frame, curves[0], curves[1])

        return img_
