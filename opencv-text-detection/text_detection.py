# USAGE
# python text_detection.py --image images/bakery.jpg 

# import the necessary packages
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import math
import argparse
import time
import cv2 as cv
import glob
import pytesseract


class OCR():
	def __init__(self, string_dictionary):
		self.folder_dir = 'images/'
		self.image_names = [file for file in glob.glob(self.folder_dir + '*JPG')]
		self.string_dictionary = string_dictionary
		self.counter = 0
		self.padding = 0.1
		self.stringThreshold = 0.7
		self.confThreshold = 0.5
		self.nmsThreshold = 0.4
		self.inpWidth = 1408
		self.inpHeight = 1056
		self.model = 'frozen_east_text_detection.pb'


	def decode(self, scores, geometry, scoreThresh):
		detections = []
		confidences = []

		# CHECK DIMENSIONS AND SHAPES OF geometry AND scores 
		assert len(scores.shape) == 4, "Incorrect dimensions of scores"
		assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
		assert scores.shape[0] == 1, "Invalid dimensions of scores"
		assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
		assert scores.shape[1] == 1, "Invalid dimensions of scores"
		assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
		assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
		assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
		height = scores.shape[2]
		width = scores.shape[3]
		for y in range(0, height):

			# Extract data from scores
			scoresData = scores[0][0][y]
			x0_data = geometry[0][0][y]
			x1_data = geometry[0][1][y]
			x2_data = geometry[0][2][y]
			x3_data = geometry[0][3][y]
			anglesData = geometry[0][4][y]
			for x in range(0, width):
				score = scoresData[x]

				# If score is lower than threshold score, move to next x
				if(score < scoreThresh):
					continue

				# Calculate offset
				offsetX = x * 4.0
				offsetY = y * 4.0
				angle = anglesData[x]

				# Calculate cos and sin of angle
				cosA = math.cos(angle)
				sinA = math.sin(angle)
				h = x0_data[x] + x2_data[x]
				w = x1_data[x] + x3_data[x]

				# Calculate offset
				offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

				# Find points for rectangle
				p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
				p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
				center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
				detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
				confidences.append(float(score))

		# Return detections and confidences
		return [detections, confidences]


	def rectangle_formation(self, vertices, origW, origH):
		startX, endX = min(vertices[0][0], vertices[1][0]), max(vertices[2][0], vertices[3][0])
		startY, endY = min(vertices[1][1], vertices[2][1]), max(vertices[0][1], vertices[3][1])
		
		width, height = np.linalg.norm(vertices[0]-vertices[1]), np.linalg.norm(vertices[1]-vertices[2])
		angle = np.arctan((vertices[2][1]-vertices[1][1])/(vertices[2][0]-vertices[1][0]))
		
		dX = (endX - startX) * self.padding
		dY = (endY - startY) * self.padding
		
		startX = int(max(0, startX - dX))
		startY = int(max(0, startY - dY))
		endX = int(min(origW, endX + (dX * 2)))
		endY = int(min(origH, endY + (dY * 2)))

		return [angle, width, height, startX, endX, startY, endY]


	def fuzzy_string_match(self, string_detected, string_desired=''):
		if not string_detected or string_detected == '':
			return False
		else:
			string_detected, string_desired = string_detected.lower(), string_desired.lower()
			# Initialize matrix of zeros
			rows, cols = len(string_detected)+1, len(string_desired)+1
			distance = np.zeros((rows,cols),dtype = int)

			# Populate matrix of zeros with the indeces of each character of both strings
			for i in range(1, rows):
				for k in range(1,cols):
					distance[i][0] = i
					distance[0][k] = k

			# Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
			for col in range(1, cols):
				for row in range(1, rows):
					if string_detected[row-1] == string_desired[col-1]:
						cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
					else:
						cost = 2
					distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
										distance[row][col-1] + 1,          # Cost of insertions
										distance[row-1][col-1] + cost)     # Cost of substitutions

			ratio = ((len(string_detected)+len(string_desired)) - distance[row][col]) / (len(string_detected)+len(string_desired))
			print('Detected text: "{}", {} from the desired "{}"'.format(string_detected, ratio, string_desired))
			if ratio >= self.stringThreshold:
				return True
			else:
				return False


	def processed_roi(self, image, angle, width, height):
		''' 
		Rotates OpenCV image around center with angle theta (in deg)
		then crops the image according to width and height.
		'''
		height, width = int(width*(1+self.padding)), int(height*(1+self.padding))
		angle *= 180/np.pi
		# grab the dimensions of the image and then determine the
		# center
		(h, w) = image.shape[:2]
		(cX, cY) = (w // 2, h // 2)
	
		# grab the rotation matrix (applying the negative of the
		# angle to rotate clockwise), then grab the sine and cosine
		# (i.e., the rotation components of the matrix)
		M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
		cos = np.abs(M[0, 0])
		sin = np.abs(M[0, 1])
	
		# compute the new bounding dimensions of the image
		nW = int((h * sin) + (w * cos))
		nH = int((h * cos) + (w * sin))
	
		# adjust the rotation matrix to take into account translation
		M[0, 2] += (nW / 2) - cX
		M[1, 2] += (nH / 2) - cY
		
		image = cv.warpAffine(image, M, (nW, nH))
		# cv.imwrite('roi-rotated.png',image)
		(cX, cY) = (nW / 2, nH / 2)

		x = int( cX - width/2  )
		y = int( cY - height/2 )
		image = image[ y:y+height, x:x+width ]
		# cv.imwrite('roi-cropped.png',image)
		return image
 

	def text_detection(self, image_name):

		# Load network
		net = cv.dnn.readNet(self.model)
		layers = []
		layers.append("feature_fusion/Conv_7/Sigmoid")
		layers.append("feature_fusion/concat_3")

		# Open a video file or an image file or a camera stream
		print("\n\n----------------Processing image", image_name, '----------------\n')
		cap = cv.VideoCapture(image_name)

		# Read frame
		hasFrame, frame = cap.read()

		# Get frame height and width
		height_ = frame.shape[0]
		width_ = frame.shape[1]
		rW = width_ / float(self.inpWidth)
		rH = height_ / float(self.inpHeight)

		# Create a 4D blob from frame.
		blob = cv.dnn.blobFromImage(frame, 1.0, (self.inpWidth, self.inpHeight), (123.68, 116.78, 103.94), True, False)

		# Run the model
		net.setInput(blob)
		outs = net.forward(layers)
		t, _ = net.getPerfProfile()

		# Get scores and geometry
		scores = outs[0]
		geometry = outs[1]
		[boxes, confidences] = self.decode(scores, geometry, self.confThreshold)

		# Apply NMS
		indices = cv.dnn.NMSBoxesRotated(boxes, confidences, self.confThreshold,self.nmsThreshold)
		parallelograms, matches, texts, text_locations = {}, {}, {}, {}
		for i in indices:
			# get 4 corners of the parallelogram and the rectangle containing it
			vertices = cv.boxPoints(boxes[i[0]])
			# scale the bounding box coordinates based on the respective ratios
			for j in range(4):
				vertices[j][0] *= rW
				vertices[j][1] *= rH
			parallelograms[str(i)] = vertices

			[angle, width, height, startX, endX, startY, endY] = self.rectangle_formation(vertices, width_, height_)
			text_locations[str(i)] = (startX, startY - 20)
			# extract the actual padded ROI
			roi = frame[startY:endY, startX:endX]
			# cv.imwrite('roi-original.png',roi)
			roi_rotated = self.processed_roi(roi, angle, width, height)

			# in order to apply Tesseract v4 to OCR text we must supply
			# (1) a language, (2) an OEM flag of 4, indicating that the we
			# wish to use the LSTM neural net model for OCR, and finally
			# (3) an OEM value, in this case, 7 which implies that we are
			# treating the ROI as a single line of text
			config = ("-l eng --oem 1 --psm 7")
			text_detected = pytesseract.image_to_string(roi_rotated, config=config)
			
			matches[str(i)] = False
			for text_desired in self.string_dictionary:
				text_matchable = self.fuzzy_string_match(text_detected, text_desired)
				if text_matchable:
					text_coordinates = (int(0.5*(startX+endX)), int(0.5*(startY+endY)))
					print('Text "{}" detected at coordinate {}\n'.format(text_desired, text_coordinates)) 
					matches[str(i)] = True
					texts[str(i)] = text_desired
					self.string_dictionary[text_desired].append((image_name.split('/')[-1], text_coordinates))
				else:
					texts[str(i)] = text_detected

		for i in indices:
			vertices = parallelograms[str(i)]
			if matches[str(i)] == False: 
				color = (0, 200, 200)
			else:
				color = (0, 0, 255)
			for j in range(4):
				p1 = (vertices[j][0], vertices[j][1])
				p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
				cv.line(frame, p1, p2, color, 3)
			cv.putText(frame, texts[str(i)], text_locations[str(i)], cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

		cv.imwrite('{}results/{}_result.png'.format(self.folder_dir, image_name.split('/')[-1][:-4]),frame)
		
		
if __name__ == "__main__":
	string_dictionary = {'Rotisserie':[], 'Cinnamon':[], 'Produce':[], 'Baking':[], 'Juices':[], 'Soda':[], 'Texas':[], 'Hot Pocket':[], 'Sara Fee':[]} # string_desired: [(image_name, coordinate)]	
	
	test = OCR(string_dictionary)
	for image_name in test.image_names:
		test.text_detection(image_name)
	print('\n==================================================\nOCR results:', test.string_dictionary, '\n')

	results = test.string_dictionary
	for key in results:
		print(key, ': ==============================')
		for item in results[key]:
			print(item)
		print('\n')

'''
==================================================
OCR results: {'Rotisserie': [('G0010284.JPG', (3357, 1300)), ('G0011256.JPG', (1772, 1185)), ('G0010473.JPG', (2556, 1349))], 'Cinnamon': [('G0010473.JPG', (1992, 1991)), ('G0010473.JPG', (2394, 1993)), ('G0010473.JPG', (1909, 1810))], 'Produce': [('G0010453.JPG', (534, 764)), ('G0010483.JPG', (2643, 895)), ('G0010697.JPG', (2258, 1045)), ('G0010342.JPG', (1296, 1042)), ('G0011471.JPG', (901, 1384)), ('G0010417.JPG', (1069, 1041)), ('G0011583.JPG', (781, 1112))], 'Baking': [('G0012121.JPG', (597, 898)), ('G0010473.JPG', (850, 1057))], 'Juices': [('G0011924.JPG', (2750, 913))], 'Soda': [('G0012121.JPG', (486, 1731)), ('G0011112.JPG', (2687, 1059))], 'Texas': [('G0011465.JPG', (2606, 953)), ('G0011465.JPG', (1909, 1142)), ('G0011471.JPG', (3158, 1152)), ('G0011863.JPG', (2403, 1304)), ('G0011863.JPG', (2138, 1316))], 'Hot Pocket': [('G0011373.JPG', (1915, 1452)), ('G0011373.JPG', (2062, 1451)), ('G0011465.JPG', (3280, 1984))], 'Sara Fee': []} 

Rotisserie : ==============================
('G0010284.JPG', (3357, 1300))
('G0011256.JPG', (1772, 1185))
('G0010473.JPG', (2556, 1349))


Cinnamon : ==============================
('G0010473.JPG', (1992, 1991))
('G0010473.JPG', (2394, 1993))
('G0010473.JPG', (1909, 1810))


Produce : ==============================
('G0010453.JPG', (534, 764))
('G0010483.JPG', (2643, 895))
('G0010697.JPG', (2258, 1045))
('G0010342.JPG', (1296, 1042))
('G0011471.JPG', (901, 1384))
('G0010417.JPG', (1069, 1041))
('G0011583.JPG', (781, 1112))


Baking : ==============================
('G0012121.JPG', (597, 898))
('G0010473.JPG', (850, 1057))


Juices : ==============================
('G0011924.JPG', (2750, 913))


Soda : ==============================
('G0012121.JPG', (486, 1731))
('G0011112.JPG', (2687, 1059))


Texas : ==============================
('G0011465.JPG', (2606, 953))
('G0011465.JPG', (1909, 1142))
('G0011471.JPG', (3158, 1152))
('G0011863.JPG', (2403, 1304))
('G0011863.JPG', (2138, 1316))


Hot Pocket : ==============================
('G0011373.JPG', (1915, 1452))
('G0011373.JPG', (2062, 1451))
('G0011465.JPG', (3280, 1984))


Sara Fee : ==============================

'''