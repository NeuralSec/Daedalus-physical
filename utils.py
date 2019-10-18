import numpy as np
import cv2

def process_image(img):
	"""
	Resize, reduce and expand image.
	# Argument:
		img: original image.

	# Returns
		image: ndarray(64, 64, 3), processed image.
	"""
	image = cv2.resize(img, (416, 416),
					   interpolation=cv2.INTER_CUBIC)
	image = np.array(image, dtype='float32')
	image /= 255.
	image = np.expand_dims(image, axis=0)
	return image

def vid2imgs(name):
	cap = cv2.VideoCapture(name)
	imgs = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not frame is None:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB
			imgs.append(process_image(frame))
		else:
			break
	cap.release()
	return np.concatenate(imgs, axis=0)