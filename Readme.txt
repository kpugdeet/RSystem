There are 670 Users and 10,000 Movies

1.Extract the zip file to Home directory

2.Install Anaconda for python 2.7
	Use pip in Anaconda to install
	- stomp.py
	- protobuf

3.Run Main.py to receive reqeust message
	python Main.py IP PORT
	Example: python Main.py 10.1.175.109 61613
	Note: IP and PORT is for the ActiveMQ Broker

4.Run Return.py to receive response from server
	python Main.py IP PORT
	Example: python Return.py 10.1.175.109 61613

5.Run Request.py to send message to server
	python Main.py IP PORT
	Example: python Request.py 10.1.175.109 61613
	Program will ask to enter command:
		queryUser (User-User recommendataion)
		queryDocument (Doc-Doc recommendation)
		querySimilarDocument (Doc topic modeling)
		updateUser 
		updateDocument
		retrain
	In each Command it will ask to enter ID:
	- queryCommand type ID that u want to query.
		Example: 1150
	- updateCommand type UserID/DocumentID and list of ID that u want to update
		Example UserID: 1150
		Example DocumentListID: 1123,2::2039,4::122,5
	- retrain will retrain all the model (userRBM, documentRBM, RSM Topic Modeling)
