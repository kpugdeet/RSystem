##################################################################
# Date    : 2016-11-21											
# Author  : Krittaphat Pugdeethosapol (krittaphat.pug@gmail.com)
# Version : 1.0													
##################################################################
import tmdbsimple as tmdb
import time
import re
import os
import string
import pickle
import json, requests

url = 'http://www.omdbapi.com/'
inputFiles = open('./OMDB_title.txt')
output = {'key':'value'}
outputFile = open('./tmp.txt','w')

PATTERN = re.compile(r'''((?:[^,"]|"[^"]*")+)''')

count = 0
inputLines = iter(inputFiles)
for lineNum, line in enumerate(inputLines):
	movieName = re.split('\(|,|:', line)[0]
	print(movieName)
	parameters = dict(plot='full',t=movieName)
	resp = requests.get(url=url, params=parameters)
	data = json.loads(resp.text)
	if data['Response'] == 'True':
		outputFile.write(data['imdbID'].encode('utf-8'))
	else:
		outputFile.write(movieName)
	outputFile.write('\n')
	time.sleep(0.25)
outputFile.close()
print(count)

	





