import cv2, sys
from skimage.feature import local_binary_pattern
import numpy as np

class LBP:
	def __init__(self, num_points=8, radius=3, eps=1e-6, resize=100):
		self.num_points = num_points * radius
		self.radius = radius
		self.eps = eps
		self.resize=resize

	def extract(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))

		lbp = local_binary_pattern(img, self.num_points, self.radius)

		return lbp.ravel()

	def extract_hist(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))

		lbp = local_binary_pattern(img, self.num_points, self.radius)

		hist = []
		window_size = 20
		for r in range(0, lbp.shape[0], window_size):
			for c in range(0, lbp.shape[1], window_size):
				window = lbp[r:r + window_size, c:c + window_size]
				hist_window, _ = np.histogram(window, bins=8)
				hist = hist + hist_window.tolist()
		norm = np.linalg.norm(hist)
		normal_array = np.array(hist) / norm
		return normal_array.tolist()

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = Extractor()
	features = extractor.extract(img)
	print(features)