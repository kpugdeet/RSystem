from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize as wordpunct_tokenize
import re
import string
import numpy
import os

INPUT_DATA_PATH = os.path.dirname(__file__) + "/../data/topicRBM/input/"

#self-defined stop words list
def get_stopwords():
    return [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours',
            u'yourself',
            u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its',
            u'itself',
            u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this',
            u'that',
            u'these',
            u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had',
            u'having', u'do',
            u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as',
            u'until',
            u'while',
            u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during',
            u'before',
            u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over',
            u'under',
            u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all',
            u'any', u'both',
            u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own',
            u'same', u'so',
            u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now']

def stemmed_words(d):
    stemmer = PorterStemmer()
    attribute_names = [stemmer.stem(token.lower()) for token in wordpunct_tokenize(
        re.sub('[%s]' % re.escape(string.punctuation), '', d)) if
                       token.lower() not in get_stopwords()]
    return attribute_names

def get_bag_words_matirx(data_path, max_vocaulary=None):
    train_data=[]
    with open(data_path,"rb") as file:
        for line in file:
            line = line.rstrip('\n')
            train_data.append(line)
    vectorizer = CountVectorizer(tokenizer=stemmed_words,max_features=max_vocaulary)
    train = vectorizer.fit_transform(train_data)
    train_data = numpy.array(train.toarray())
    vocabulary = []
    for word in vectorizer.vocabulary_:
        vocabulary.append(str(word))
    vocabulary.sort()
    with open(INPUT_DATA_PATH+"OMDB_vocabulary.dat", "wb") as file:
        numpy.savez(file, vocabulary=vocabulary)
    with open(INPUT_DATA_PATH+"OMDB.dat", "wb") as file:
        numpy.savez(file, train_data=train_data)
    return train_data

def add_new_bag_words_matirx(new_doc_path, new_doc_id, max_vocaulary=None):
    new_doc_data=[]
    new_doc_title=[]
    new_doc_tag=[]
    with open(new_doc_path, "rb") as file:
        for line in file:
            line = line.rstrip('\n')
            tmp=line.split("::")

            new_doc_title.append(tmp[0])
            new_doc_tag.append(tmp[1])
            new_doc_data.append(tmp[2])
    vocabulary=numpy.load(INPUT_DATA_PATH+"OMDB_vocabulary.dat")["vocabulary"]
    vectorizer = CountVectorizer(vocabulary=vocabulary,tokenizer=stemmed_words,max_features=max_vocaulary)
    new_doc_maxtirx=vectorizer.transform(new_doc_data)
    new_doc_maxtirx = numpy.array(new_doc_maxtirx.toarray())
    old_train_data=numpy.load(INPUT_DATA_PATH+"OMDB.dat")["train_data"]
    new_data=numpy.concatenate([old_train_data,new_doc_maxtirx])
    with open(INPUT_DATA_PATH+"OMDB.dat", "wb") as file:
        numpy.savez(file, train_data=new_data)
    #update id  
    data_ID=load_movie_ID()
    data_ID.append(str(new_doc_id))
    #update title
    data_title=load_movie_title()
    data_title.append(new_doc_title[0])
    #update tag
    data_tag=load_movie_tag()
    data_tag.append(new_doc_tag[0].split("|"))
    #update_overview
    data_overview=load_movie_overview()
    data_overview.append(new_doc_data[0])

    return new_doc_maxtirx, new_data, data_ID,data_title,data_tag, data_overview

def load_movie_overview():
    filename = INPUT_DATA_PATH+"OMDB_dataset_with_stopword.txt"
    data = []
    with open(filename, "rb") as file:
        for line in file:
            line = line.rstrip('\n')
            data.append(line)
    return data

def load_movie_title():
    filename = INPUT_DATA_PATH+"OMDB_title.txt"
    title_data = []
    with open(filename, "rb") as file:
        for line in file:
            line = line.rstrip('\n')
            title_data.append(line)
    return title_data

def load_movie_tag():
    filename=INPUT_DATA_PATH+"OMDB_tag.txt"
    tag_data = []
    with open(filename, "rb") as file:
        for line in file:
            line = line.rstrip('\n')
            tag_data.append(line.split("|"))
    return tag_data

def load_movie_ID():
    filename=INPUT_DATA_PATH+"OMDB_movieID.txt"
    ID_data = []
    with open(filename, "rb") as file:
        for line in file:
            line = line.rstrip('\n')
            ID_data.append(line)
    return ID_data
