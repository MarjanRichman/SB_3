import math
import numpy as np

class Evaluation:

	def compute_rank1(self, Y, y):
		classes = np.unique(sorted(y))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			idx1 = y==cla1
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = Y[idx1==True, :]
			Y1[Y1==0] = math.inf
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0]
				imin = idx1[smin]
				count_all += 1
				if imin:
					count_correct += 1
		return count_correct/count_all*100


	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...

	def compute_rank5(self, Y, y):
		classes = np.unique(sorted(y))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			idx1 = y == cla1
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = Y[idx1 == True, :]
			Y1[Y1 == 0] = math.inf
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0:5]
				imin = idx1[smin]
				count_all += 1
				if imin.any():
					count_correct += 1
		return count_correct / count_all * 100
