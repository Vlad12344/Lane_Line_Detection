import cv2
import numpy as np

from config import Config

class Birdeye_view(Config):

	def __init__(self):
		self.M = cv2.getPerspectiveTransform(self.src, self.dst)
		self.inv_M = cv2.getPerspectiveTransform(self.dst, self.src)

	def pipeline(self, img, s_thresh=(100, 255), sx_thresh=(15, 255)):
		# Highlight frame area 
		img = img[self.left_up_corner_point:(self.left_up_corner_point + self.IMG_H), 0:self.IMG_W]
		
		# Convert to HLS color space and separate the V channel
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)    
		h_channel = hls[:,:,0]
		l_channel = hls[:,:,1]
		s_channel = hls[:,:,2]

		# Sobel x
		# Take the derivative in x
		sobel_l_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)

		# Absolute x derivative to accentuate lines away from horizontal
		abs_sobelx = np.absolute(sobel_l_x) 
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

		# Threshold x gradient
		sxbinary = np.zeros_like(scaled_sobel)
		sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

		# Threshold color channel
		s_binary = np.zeros_like(l_channel)
		s_binary[(l_channel >= s_thresh[0]) & (l_channel <= s_thresh[1])] = 1

		color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
		combined_binary = np.zeros_like(sxbinary)
		combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

		return combined_binary

	def perspective_warp(self, img):
		# Warp the image using OpenCV warpPerspective()        
		warped_img = cv2.warpPerspective(img, self.M, (self.IMG_W, self.IMG_H))
		# Highlights the needed lane are. Remaining areas are becomes zeros.
		warped_img = cv2.bitwise_and(warped_img, warped_img, mask=self.mask)

		return warped_img

	def inv_perspective_warp(self, img):
		# Warp the image using OpenCV warpPerspective()        
		return cv2.warpPerspective(img, self.inv_M, (self.IMG_W, self.IMG_H))

	def get_perspective_warp(self, img):
		binary_img = self.pipeline(img)
		return self.perspective_warp(binary_img)