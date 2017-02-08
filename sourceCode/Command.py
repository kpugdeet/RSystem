##################################################################
# Date    : 2016-11-17											
# Author  : Krittaphat Pugdeethosapol (krittaphat.pug@gmail.com)
# Update  : 2016-12-13
# Version : 3.0									
##################################################################

import sys
import time
import stomp
import os
import pickle
import traceback
import RecommenderSystem_pb2
from RunRBM import RunRBM
from RSM import RSM

baseFolder = '../data/cfRBM/'

# Command Class
class Command:
	def __init__(self):
		# Load previous result object
		try:
			self.userFileResult = pickle.load(open(baseFolder + 'UserOutput.dat', 'rb'))
			self.docFileResult = pickle.load(open(baseFolder + 'DocOutput.dat', 'rb'))
			self.docSimFileResult = pickle.load(open('../data/topicRBM/output/Result.dat', 'rb'))
		except:
			self.userFileResult = dict()
			self.docFileResult = dict()
			self.docSimFileResult = dict()

		self.userRBM = RunRBM(self.userFileResult, baseFolder + 'UserInfo.dat', baseFolder + 'UserOutput.dat', 'UserRBMK', baseFolder + 'userKMap.object')
		self.docRBM = RunRBM(self.docFileResult, baseFolder + 'DocInfo.dat', baseFolder + 'DocOutput.dat', 'DocRBMK', baseFolder + 'docKMap.object')
		self.docRSM = RSM('OMDB_dataset_without_stopword.txt')

	# Print Info
	def printLog (self, status, command, idStr, output, unique):
		print('{0:20}{1:7}{2:25}{3:15}{4}'.format(unique, status, command, idStr, output))

	# Query User with ID
	def queryUser (self, requestMsg):
		# Generate Return Message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'returnUser'
		returnMsg.userID = requestMsg.userID
		returnMsg.unique = requestMsg.unique

		# Get Result
		try:
			returnMsg.documentID = self.userFileResult[requestMsg.userID]
			self.printLog('INFO', 'queryUser', requestMsg.userID, returnMsg.documentID, returnMsg.unique)
		except:
			returnMsg.documentID = 'UserID Not Found'
			self.printLog('ERROR', 'queryUser', requestMsg.userID, returnMsg.documentID, returnMsg.unique)

		return  returnMsg

	# Query Document
	def queryDocument (self, requestMsg):
		# Generate Return Message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'returnDocument'
		returnMsg.documentID = requestMsg.documentID
		returnMsg.unique = requestMsg.unique

		# Get Result
		try:
			returnMsg.userID = self.docFileResult[requestMsg.documentID]
			self.printLog('INFO', 'queryDocument', requestMsg.documentID, returnMsg.userID, returnMsg.unique)
		except:
			returnMsg.userID = 'DocumentID Not Found'
			self.printLog('ERROR', 'queryDocument', requestMsg.documentID, returnMsg.userID, returnMsg.unique)

		return returnMsg

	# Query Similar Document
	def querySimilarDocument (self, requestMsg):
		# Generate Return Message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'returnSimilarDocument'
		returnMsg.documentID = requestMsg.documentID
		returnMsg.unique = requestMsg.unique

		# Get Result
		try:
			returnMsg.userID = self.docSimFileResult[requestMsg.documentID]
			self.printLog('INFO', 'querySimilarDocument', requestMsg.documentID, returnMsg.userID, returnMsg.unique)
		except:
			returnMsg.userID = 'DocumentID Not Found'
			self.printLog('ERROR', 'querySimilarDocument', requestMsg.documentID, returnMsg.userID, returnMsg.unique)

		return returnMsg

	# Update User
	def updateUser (self, requestMsg):
		# Generate Return Message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'status'
		returnMsg.userID = requestMsg.userID
		returnMsg.unique = requestMsg.unique

		try:
			# Update in UserInfo.dat
			if (requestMsg.documentID):
				found = 0
				with open(baseFolder+'UserInfo.dat') as oldfile, open(baseFolder+'UserInfoTmp.dat', 'w') as newfile:
					for line in oldfile:
						if line.startswith(requestMsg.userID + '::'):
							newfile.write(line[:-1] + '::' + requestMsg.documentID + '\n')
							userDocumentID = line[len(requestMsg.userID + '::'):-1] + '::' + requestMsg.documentID
							found = 1
						else:
							newfile.write(line)
							userDocumentID = requestMsg.documentID
					# Append to file
					if not found:
						newfile.write('{0}::{1}\n'.format(requestMsg.userID,requestMsg.documentID))
				os.system('mv '+baseFolder+'UserInfoTmp.dat '+baseFolder+'UserInfo.dat')

			# Get new result for update User
			self.userFileResult = self.userRBM.updateOutput(requestMsg.userID, userDocumentID)

			returnMsg.status = 'user success'
			self.printLog('INFO', 'updateUser', requestMsg.userID, returnMsg.status, returnMsg.unique)
	
		except Exception as e: 
			traceback.print_exc()
			returnMsg.status = 'user fail'
			self.printLog('ERROR', 'updateUser', requestMsg.userID, returnMsg.status, returnMsg.unique)

		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'returnUser'
		returnMsg.userID = requestMsg.userID
		returnMsg.unique = requestMsg.unique

		# Get Result
		try:
			returnMsg.documentID = self.userFileResult[requestMsg.userID]
			self.printLog('INFO', 'queryUser', requestMsg.userID, returnMsg.documentID, returnMsg.unique)
		except:
			returnMsg.documentID = 'UserID Not Found'
			self.printLog('ERROR', 'queryUser', requestMsg.userID, returnMsg.documentID, returnMsg.unique)

		return returnMsg

	# Update Document
	def updateDocument (self, requestMsg):
		# Generate Return Message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'status'
		returnMsg.documentID = requestMsg.documentID
		returnMsg.unique = requestMsg.unique
		documentUserID = ''

		try:
			# Update document data
			found = 0
			with open(baseFolder + 'DocInfo.dat') as oldfile, open(baseFolder + 'DocInfoTmp.dat', 'w') as newfile:
				for line in oldfile:
					if line.startswith(requestMsg.documentID + '::'):
						if requestMsg.userID:
							newfile.write(line[:-1] + '::' + requestMsg.userID + '\n')
							documentUserID = line[len(requestMsg.documentID + '::'):-1] + '::' + requestMsg.userID
						else:
							documentUserID = line[len(requestMsg.documentID + '::'):-1]
						found = 1
					else:
						newfile.write(line)
						if requestMsg.userID and found == 0:
							documentUserID = requestMsg.userID
				if not found and requestMsg.userID:
					newfile.write('{0}::{1}\n'.format(requestMsg.documentID, requestMsg.userID))
			os.system('mv ' + baseFolder + 'DocInfoTmp.dat ' + baseFolder + 'DocInfo.dat')
			
			# Update text in Document
			if (requestMsg.documentTitle and requestMsg.documentText):
				with open('../data/topicRBM/input/' + requestMsg.documentID + '.dat', 'w') as myfile:
					myfile.write('{0}::{1}::{2}'.format(requestMsg.documentTitle, 'drama|action', requestMsg.documentText))
				# with open(baseFolder + 'topicRBM/input/movielens_dataset.txt', 'a') as myfile:
				# 	myfile.write('{0}\n'.format(requestMsg.documentText))
			
			# Update for new document
			self.docSimFileResult = self.docRSM.update(requestMsg.documentID + '.dat', requestMsg.documentID)

			# Get new result for update User and doing Ad-Hoc
			if documentUserID != '':
				self.docFileResult = self.docRBM.updateOutput(requestMsg.documentID, documentUserID)
			else:
				self.docFileResult = self.docRBM.updateOutput(requestMsg.documentID, documentUserID)
				# self.docFileResult = self.docRBM.updateAdHoc(requestMsg.documentID, topUserID[requestMsg.documentID])

			returnMsg.status = 'doc success'
			self.printLog('INFO', 'updateDocument', requestMsg.documentID, returnMsg.status, returnMsg.unique)

		except Exception as e:
			traceback.print_exc()
			returnMsg.status = 'doc fail'
			self.printLog('ERROR', 'updateDocument', requestMsg.documentID, returnMsg.status, returnMsg.unique)

		# Generate Return Message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'returnSimilarDocument'
		returnMsg.documentID = requestMsg.documentID
		returnMsg.unique = requestMsg.unique

		# Get Result
		try:
			returnMsg.userID = self.docSimFileResult[requestMsg.documentID]
			self.printLog('INFO', 'querySimilarDocument', requestMsg.documentID, returnMsg.userID, returnMsg.unique)
		except:
			returnMsg.userID = 'DocumentID Not Found'
			self.printLog('ERROR', 'querySimilarDocument', requestMsg.documentID, returnMsg.userID, returnMsg.unique)
		
		return returnMsg

	# Retrain
	def retrain (self, requestMsg):
		# Generate Return Message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'status'
		returnMsg.unique = requestMsg.unique

		try:
			# Train userRBM
			self.userFileResult = self.userRBM.retrain()
			# Train docRBM
			self.docFileResult = self.docRBM.retrain()
			# Train docRSM
			self.docSimFileResult = self.docRSM.retrain()

			self.printLog('INFO', 'command', '', 'retrain success', returnMsg.unique)
			returnMsg.status = 'retrain success'
		except Exception as e:
			traceback.print_exc()
			returnMsg.status = 'retrain fail'

		return returnMsg

	# Retrain
	def error (self, requestMsg):
		# Generate Return Message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.command = 'status'
		returnMsg.unique = requestMsg.unique
		returnMsg.status = 'command error'
		self.printLog('ERROR', 'command', '', returnMsg.status, returnMsg.unique)

		return returnMsg

