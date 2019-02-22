import numpy as np

class Config:
	"""
		IMG_H and IMG_W are parameters needed for cutting window from video frame.
		
		IMG_H - height of cutted window
		IMG_W - width of cutted window
	"""
	IMG_H = 220
	IMG_W = 1280
	"""
		This parameters are uses in sliding_window function.

		minpix - minimal amount of pixels, which must be in sliding 
				 window to recalculate shifting parameter of window
		margin - half sliding window width(megered in pixels). 
				 We are use half because know about mid point of sliding window.
				 20 pxl is set aside on each side to get corner points of sliding window.
		nwindows - amount of sliding windows 
	"""
	minpix = 1
	margin = 20
	nwindows = 4
	"""
		left_up_corner_point - point of cutted window from video frame. 
							   Must be <= 0 but >= (frame height - IMG_H).
	"""
	left_up_corner_point = 360
	"""
		This parameter is uses in perspective_warp function in bitwice operation.

		Highlights the needed lane are. Remaining areas are becomes zeros. 
	"""
	mask = np.zeros((IMG_H, IMG_W), dtype='uint8')
	mask[0:IMG_H, 420:600] = 255
	"""
		Parameters are uses for finding transformation matrix and inferse transformation matrix.

		src - 4 points on the input image 
		dst - 4 corresponding points on the output image
	"""
	src = np.float32([[0, IMG_H], [1207, IMG_H], [0, 0], [IMG_W, 0]])
	dst = np.float32([[450, IMG_H], [600, IMG_H], [0, 0], [IMG_W, 0]])
	"""
		video_path - path to the source video
	"""
	video_path = '/media/vlados/FreeSpace/CV&NN/CV Project/New Project.mp4'