##################################################################
# Date    : 2016-11-17											
# Author  : Krittaphat Pugdeethosapol (krittaphat.pug@gmail.com)
# Update  : 2016-12-13
# Version : 3.0												
##################################################################

import numpy as np
import math
import time
import datetime
import ntpath
import pickle
import os
import traceback
import sys
import RecommenderSystem_pb2
from RBMK import RBMK
import collections

class RunRBM:
	def __init__(self, outputArray, inputFile, outputFile, typeRBM, mapping):
		self.outputArray = outputArray
		self.inputFile = inputFile
		self.outputFile = outputFile
		self.mapFile = mapping
		self.rbm = RBMK ('../data/Config.ini', typeRBM)

		# Initial the mapping ID
		try:
			self.mapping = pickle.load(open(self.mapFile, 'rb' ))
		except Exception as e:
			traceback.print_exc()
			sys.exit()

	# Retrain model
	def retrain (self):
		startTime = time.time()

		# ReadFile
		print('{0:7}Loading data from {1}'.format('INFO',self.inputFile))
		filePointer = open(self.inputFile)
		iterLines = iter(filePointer)
		dataID = []
		data = [[] for x in range(self.rbm.k)]
		for lineNum, line in enumerate(iterLines):
			tmp = [ [0 for i in range(self.rbm.numVisible) ] for j in range(self.rbm.k) ]
			ID = line.split('::')[0]
			line = line.split('::')[1:]
			exID = np.random.randint((len(line)/4)+1)
			for offset, ele in enumerate(line):
				try:
					idTmp = ele.split(',')[0]
					rate = ele.split(',')[1]
					tmp[int(float(rate))][int(idTmp)] = int(1)
					if offset == exID:
						# print('Ex {0} {1}'.format(lineNum,offset))
						tmp[int(float(rate))][int(idTmp)] = int(0)
						countExclude += 1
				except:
					exID += 1
					tmpFalse = None
			for i in range(self.rbm.k):
				data[i].append(tmp[i])
			dataID.append(ID)
		data = np.array(data)

		# Training
		print('{0:7}Start training RBM {1}'.format('INFO', self.inputFile))
		# trainPart = 0.8
		# trainSize = int(trainPart * len(data))
		# train = np.array(data[:trainSize])
		# test = np.array(data[trainSize:])
		# self.rbm.train (train, test)
		self.rbm.train (data)
		totalTime = time.time() - startTime
		print ('{0:7}Finish training RBM {1} Time : {2}'.format('INFO', self.inputFile, totalTime))

		# Calculate all output
		tmpHidden = self.rbm.getHidden(data)
		tmpVisible = self.rbm.getVisible(tmpHidden)
		tmpVisible = np.array(tmpVisible)

		# Calculate Output to select right pos for each visble unit
		output = [[None for i in range(tmpVisible.shape[2])] for j in range(tmpVisible.shape[1])]
		for userID in range(tmpVisible.shape[1]):
			for docID in range(tmpVisible.shape[2]):
				maxValue = tmpVisible[0][userID][docID]
				maxPos = 0
				for dim in range(1,self.rbm.k):
					if (tmpVisible[dim][userID][docID] > tmpVisible[dim-1][userID][docID]):
						maxValue = tmpVisible[dim][userID][docID]
						maxPos = dim
				output[userID][docID] = dict({'doc':docID, 'pos':maxPos, 'value':maxValue})

		# Save Output
		a = time.time()
		self.outputArray = {'key':'value'}
		output = np.array(output)
		for i in range(output.shape[0]):
			maxTop = 10
			countTop = 0
			tmpValue = ''
			# posPoint = 1
			# tmpList = sorted([x for x in output[i] if x['pos']==posPoint], key=lambda k:k['value'], reverse=True)
			tmpList = sorted([x for x in output[i]], key=lambda k:k['value'], reverse=True)
			for j in range(len(tmpList)):
				if self.rbm.exclude[i][int(tmpList[j]['doc'])] == 0:
					if countTop != 0:
						tmpValue += '::'
					try:
						tmpValue += self.mapping[str(tmpList[j]['doc'])]
						countTop = countTop + 1
						if countTop == maxTop:
							break
					except:
						tmpFalse = None
			self.outputArray[dataID[i]] = tmpValue

		pickle.dump(self.outputArray, open(self.outputFile,'wb'))
		print('{0:7}Output of training RBM {1} Save to {2} Time: {3}'.format('INFO', self.inputFile, self.outputFile, time.time()-a))

		# return outputArray
		return self.outputArray

	def updateOutput (self, tmpID, listID):
		# Read Data
		tmpData = [[] for x in range(self.rbm.k)]
		tmp = [ [0 for i in range(self.rbm.numVisible) ] for j in range(self.rbm.k) ]
		line = listID.split('::')
		for doc in line:
			try:
				idTmp = ele.split(',')[0]
				rate = ele.split(',')[1]
				tmp[int(float(rate))][int(self.mapping[idTmp])] = int(1)
			except:
				tmpFalse = None
		for i in range(self.rbm.k):
			tmpData[i].append(tmp[i])
		tmpData = np.array(tmpData)

		# Calculate all output
		tmpHidden = self.rbm.getHidden(tmpData)
		tmpVisible = self.rbm.getVisible(tmpHidden)
		tmpVisible = np.array(tmpVisible)

		# Calculate Output to select right pos for each visble unit
		output = [[None for i in range(tmpVisible.shape[2])] for j in range(tmpVisible.shape[1])]
		for userID in range(tmpVisible.shape[1]):
			for docID in range(tmpVisible.shape[2]):
				maxValue = tmpVisible[0][userID][docID]
				maxPos = 0
				for dim in range(1,self.rbm.k):
					if (tmpVisible[dim][userID][docID] > tmpVisible[dim-1][userID][docID]):
						maxValue = tmpVisible[dim][userID][docID]
						maxPos = dim
				output[userID][docID] = dict({'doc':docID, 'pos':maxPos, 'value':maxValue})

		# Save Output
		maxTop = 10
		countTop = 0
		tmpValue = ''
		tmpList = sorted([x for x in output[0]], key=lambda k:k['value'], reverse=True)
		for j in range(len(tmpList)):
			if countTop != 0:
				tmpValue += '::'
			try:
				tmpValue += self.mapping[str(tmpList[j]['doc'])]
				countTop = countTop + 1
				if countTop == maxTop:
					break
			except:
				tmpFalse = None
		self.outputArray[tmpID] = tmpValue
		pickle.dump(self.outputArray, open(self.outputFile,'wb'))

		# return outputArray
		return self.outputArray

	def updateAdHoc (self, tmpID, listID):
		# Read Data
		listArray = []
		tmpDoc = listID.split('::')
		for doc in tmpDoc:
			docID = doc.rpartition('(')[0]
			try:
				for userID in self.outputArray[docID].split('::'):
					listArray.append(int(userID))
			except:
				tmpFalse = None

		# Calculate and SaveOutput
		maxTop = 10
		userCountTop = 0
		userTmpValue = ''
		x = collections.Counter(listArray)
		for elt,count in x.most_common():
			userTmpValue += str(elt)
			if (userCountTop < maxTop-1):
				userTmpValue += '::'
			userCountTop = userCountTop + 1
			if userCountTop == maxTop:
				break
		self.outputArray[tmpID] = userTmpValue
		pickle.dump(self.outputArray, open(self.outputFile,'wb'))

		# return outputArray
		return self.outputArray


