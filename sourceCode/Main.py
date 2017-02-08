##################################################################
# Date    : 2016-11-17											
# Author  : Krittaphat Pugdeethosapol (krittaphat.pug@gmail.com)
# Version : 2.0									
##################################################################

import sys
import time
import stomp
import os
import RecommenderSystem_pb2
import threading
from Command import Command


# Class for stomp listener
class Listener (stomp.ConnectionListener):

	def __init__(self, sender):
		self.sender = sender
		self.command = Command()
		self.lock = threading.Lock()

	# Message receive error
	def on_error (self, headers, message):
		print ('Received an error %s' % message)

	# Message receive success
	def on_message (self, headers, message):

		# Initial Request/Return Message
		requestMsg = RecommenderSystem_pb2.Request()
		returnMsg = RecommenderSystem_pb2.Return()

		# Parse Protobuf Request message
		requestMsg.ParseFromString(message)

		# Query User
		if (requestMsg.command == 'queryUser' and requestMsg.userID):
			returnMsg = self.command.queryUser(requestMsg)

		# Query Document
		elif (requestMsg.command == 'queryDocument' and requestMsg.documentID):
			returnMsg = self.command.queryDocument(requestMsg)

		# Query Similar Document
		elif (requestMsg.command == 'querySimilarDocument' and requestMsg.documentID):
			returnMsg = self.command.querySimilarDocument(requestMsg)

		# Update User
		elif (requestMsg.command == 'updateUser' and requestMsg.userID and requestMsg.documentID):
			t =  threading.Thread(name = 'UpdateUser Thread', target = self.threadUpdateUser, args = (requestMsg, ))
			t.start()

		# Update Document
		elif (requestMsg.command == 'updateDocument' and requestMsg.documentID and (requestMsg.userID or (requestMsg.documentTitle and requestMsg.documentText))):
			t1 =  threading.Thread(name = 'UpdateDoc Thread', target = self.threadUpdateDoc, args = (requestMsg, ))
			t1.start()

		# Retrain
		elif (requestMsg.command == 'retrain'):
			t2 =  threading.Thread(name = 'Retrain Thread', target = self.threadRetrain, args = (requestMsg, ))
			t2.start()
		
		# Error	
		else:
			returnMsg = self.command.error(requestMsg)

		# Send back to client
		if (requestMsg.command != 'retrain' and requestMsg.command != 'updateUser' and requestMsg.command != 'updateDocument'):
			self.sender.send(body = returnMsg.SerializeToString(), destination = '/queue/sender')


	# UpdateUserthread
	def threadUpdateUser (self, requestMsg):
		self.lock.acquire()
		try:
			returnMsgThread = RecommenderSystem_pb2.Return()
			returnMsgThread = self.command.updateUser(requestMsg)
			self.sender.send(body = returnMsgThread.SerializeToString(), destination = '/queue/sender')
		finally:
			self.lock.release()

	# UpdateDocument thread
	def threadUpdateDoc (self, requestMsg):
		self.lock.acquire()
		try:
			returnMsgThread = RecommenderSystem_pb2.Return()
			returnMsgThread = self.command.updateDocument(requestMsg)
			self.sender.send(body = returnMsgThread.SerializeToString(), destination = '/queue/sender')
		finally:
			self.lock.release()

	# Retrain thread
	def threadRetrain (self, requestMsg):
		self.lock.acquire()
		try:
			returnMsgThread = RecommenderSystem_pb2.Return()
			returnMsgThread = self.command.retrain(requestMsg)
			self.sender.send(body = returnMsgThread.SerializeToString(), destination = '/queue/sender')
		finally:
			self.lock.release()

# Check the argument of the command
if (len(sys.argv) != 3):
	print ("Need arguments: python Main.py IP PORT USERRESULT DOCRESULT (10.35.23.2, 61613)")
	sys.exit()

# Create connection to subscribe to ActiveMQ broker IP (10.35.23.2) & Port (61613) 
connection = stomp.Connection(host_and_ports = [(sys.argv[1], sys.argv[2])])

# Set the listener function when receive message
connection.set_listener ('', Listener(connection))

# Start connection 
connection.start()
connection.connect()

# Subscribe to specific queue for receiveing message
connection.subscribe(destination = '/queue/carbon', id = 1)

print ("System start running")
print('{0:20}{1:7}{2:25}{3:15}{4}'.format('TIMESTAMP', 'TYPE', 'COMMAND', 'ID', 'OUTPUT'))
# Loop to make this main still running all time
while (1):
   time.sleep(10)
   
# Close connection
connection.disconnect()