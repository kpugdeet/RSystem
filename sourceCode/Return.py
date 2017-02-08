import sys
import time
import stomp
import random
import datetime
import RecommenderSystem_pb2

class Listener (stomp.ConnectionListener):
	# Message receive error
	def on_error (self, headers, message):
		print ('Received an error %s' % message)
	# Message receive success
	def on_message (self, headers, message):
		# Parse Protobuf Request message
		returnMsg = RecommenderSystem_pb2.Return()
		returnMsg.ParseFromString(message)

		if returnMsg.command == 'returnUser':
			print('{0:20}{1:10}{2:25}{3:20}{4:15}{5}'.format(returnMsg.unique, 'RETURN', returnMsg.command, returnMsg.status, returnMsg.userID, returnMsg.documentID))
		else:
			print('{0:20}{1:10}{2:25}{3:20}{4:15}{5}'.format(returnMsg.unique, 'RETURN', returnMsg.command, returnMsg.status, returnMsg.documentID, returnMsg.userID))

# Check the argument of the command
if (len(sys.argv) != 3):
	print ("Need arguments: python Main.py IP PORT (10.35.23.2 61613)")
	sys.exit()

print('{0:20}{1:10}{2:25}{3:20}{4:15}{5}'.format('TIMESTAMP', 'TYPE', 'COMMAND', 'STATUS', 'QUERYID', 'RESULTID'))

# Create connection to subscribe to ActiveMQ broker IP () & Port (61613) 
connection = stomp.Connection(host_and_ports=[(sys.argv[1], sys.argv[2])])

# Set the listener function when receive message
connection.set_listener ('', Listener())

# Start connection 
connection.start()
connection.connect()

# Subscribe to specific queue for receiveing message
connection.subscribe(destination='/queue/sender', id = 1)

while (1):
	time.sleep(1000)
   
# Close connection
connection.disconnect()
