# USAGE
# python text_detection.py --image images/bakery.jpg 

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import math
import argparse
import time
import cv2 as cv
import glob
import pytesseract


class OCR():
	def __init__(self):
		folder_dir = 'images/'
		self.image_names = [file for file in glob.glob(folder_dir + '*JPG')]

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


	def rectangle_formation(self, vertices, origW, origH, padding=0.05):
		startX, endX = min(vertices[0][0], vertices[1][0]), max(vertices[2][0], vertices[3][0])
		startY, endY = min(vertices[1][1], vertices[2][1]), max(vertices[0][1], vertices[3][1])
		# print('rectangle:', startX, startY, endX, endY)
		dX = (endX - startX) * padding
		dY = (endY - startY) * padding
		startX = int(max(0, startX - dX))
		startY = int(max(0, startY - dY))
		endX = int(min(origW, endX + (dX * 2)))
		endY = int(min(origH, endY + (dY * 2)))
		return startX, endX, startY, endY


	def fuzzy_string_match(self, string_detected, string_desired=''):
		if not string_detected or string_detected == '':
			return False
		else:
			return string_detected


	def text_detection(self, image_name):
		# Read and store arguments
		confThreshold = 0.5
		nmsThreshold = 0.4
		inpWidth = 800
		inpHeight = 800
		model = 'frozen_east_text_detection.pb'

		# Load network
		net = cv.dnn.readNet(model)
		layers = []
		layers.append("feature_fusion/Conv_7/Sigmoid")
		layers.append("feature_fusion/concat_3")

		# Open a video file or an image file or a camera stream
		print("Processing image", image_name)
		cap = cv.VideoCapture(image_name)

		# Read frame
		hasFrame, frame = cap.read()

		# Get frame height and width
		height_ = frame.shape[0]
		width_ = frame.shape[1]
		rW = width_ / float(inpWidth)
		rH = height_ / float(inpHeight)

		# Create a 4D blob from frame.
		blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

		# Run the model
		net.setInput(blob)
		outs = net.forward(layers)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

		# Get scores and geometry
		scores = outs[0]
		geometry = outs[1]
		[boxes, confidences] = self.decode(scores, geometry, confThreshold)

		# initialize the list of results
		results = []

		# Apply NMS
		indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
		for i in indices:
			# get 4 corners of the parallelogram and the rectangle containing it
			vertices = cv.boxPoints(boxes[i[0]])
			# scale the bounding box coordinates based on the respective ratios
			for j in range(4):
				vertices[j][0] *= rW
				vertices[j][1] *= rH
			for j in range(4):
				p1 = (vertices[j][0], vertices[j][1])
				p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
				cv.line(frame, p1, p2, (0, 255, 0), 1)

			startX, endX, startY, endY = self.rectangle_formation(vertices, width_, height_)
			cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

			# extract the actual padded ROI
			roi = frame[startY:endY, startX:endX]

			# in order to apply Tesseract v4 to OCR text we must supply
			# (1) a language, (2) an OEM flag of 4, indicating that the we
			# wish to use the LSTM neural net model for OCR, and finally
			# (3) an OEM value, in this case, 7 which implies that we are
			# treating the ROI as a single line of text
			config = ("-l eng --oem 1 --psm 7")
			text = pytesseract.image_to_string(roi, config=config)
			detected_text = self.fuzzy_string_match(text)
			if detected_text:
				text_coordinates = (int(0.5*(startX+endX)), int(0.5*(startY+endY)))
				print('Text "{}" detected at coordinate {}\n'.format(detected_text, text_coordinates)) 
				cv.putText(frame, text, (startX, startY - 20), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

		# add the bounding box coordinateś and OCR'd text to the list of results
		results.append(((startX, startY, endX, endY), text))





		# Display the frame
		# cv.imshow(kWinName,frame)
		cv.imwrite('images/result.png',frame)
		
		
if __name__ == "__main__":
	test = OCR()
	test.text_detection(test.image_names[0])