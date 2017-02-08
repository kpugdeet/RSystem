import sys
import time
import stomp
import random
import datetime
import RecommenderSystem_pb2

# Check the argument of the command
if (len(sys.argv) != 3):
	print ("Need arguments: python Main.py IP PORT (10.35.23.2 61613)")
	sys.exit()

# Create connection to subscribe to ActiveMQ broker IP () & Port (61613) 
connection = stomp.Connection(host_and_ports=[(sys.argv[1], sys.argv[2])])

# Start connection 
connection.start()
connection.connect()

while (1):
	# Create protobuf
	request = RecommenderSystem_pb2.Request()
	randomCommand = raw_input('Enter command: ')

	dateTimeString = str(datetime.datetime.now().strftime('%y%m%d%H%M%S'))
	if randomCommand == 'queryUser':
		request.command = 'queryUser'
		request.userID = raw_input('{0} Enter userID: '.format(request.command))
		print('{0:15}{1:15}{2:15}{3}'.format('REQUEST', dateTimeString, request.command, request.userID))

	elif randomCommand == 'queryDocument':
		request.command = 'queryDocument'
		request.documentID = raw_input('{0} Enter documentID: '.format(request.command))
		print('{0:15}{1:15}{2:15}{3}'.format('REQUEST', dateTimeString, request.command, request.documentID))

	elif randomCommand == 'querySimilarDocument':
		request.command = 'querySimilarDocument'
		request.documentID = raw_input('{0} Enter documentID: '.format(request.command))
		print('{0:15}{1:15}{2:15}{3}'.format('REQUEST', dateTimeString, request.command, request.documentID))

	elif randomCommand == 'updateUser':
		request.command = 'updateUser'
		request.userID = raw_input('{0} Enter userID: '.format(request.command))
		request.documentID = raw_input('{0} Enter documentID: '.format(request.command))
		print('{0:15}{1:15}{2:15}{3}'.format('UPDATE', dateTimeString, request.command, request.documentID))

	elif randomCommand == 'updateDocument':
		request.command = 'updateDocument'
		request.documentID = raw_input('{0} Enter documentID: '.format(request.command))
		request.userID = raw_input('{0} Enter userID: '.format(request.command))
		request.documentTitle = raw_input('{0} Enter Title: '.format(request.command))
		request.documentText = raw_input('{0} Enter Text: '.format(request.command))
		print('{0:15}{1:15}{2:15}{3}'.format('UPDATE', dateTimeString, request.command, request.documentID))

	elif randomCommand == 'retrain':
		request.command = 'retrain'
		print('{0:10}{1:15}{2:15}{3}'.format('RETRAIN', request.command, '', dateTimeString))

	request.unique = dateTimeString
	connection.send(body=request.SerializeToString(), destination='/queue/carbon')
   
# Close connection
connection.disconnect()
