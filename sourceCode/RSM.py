from __future__ import division
import numpy
from scipy import spatial
import data_movie
import os.path
import pickle

#Constant data path
INPUT_DATA_PATH = os.path.dirname(__file__) + "/../data/topicRBM/input/"
OUTPUT_DATA_PATH = os.path.dirname(__file__) + "/../data/topicRBM/output/"
PARAMS_DATA_PATH = os.path.dirname(__file__) + "/../data/topicRBM/params/"

class RSM:
    def __init__(
            self,
            filename,  # "../data/topicRBM/input/movieslens_dataset.txt", need to config
    ):
        self.input = self.preprocess_txt_rawdata(INPUT_DATA_PATH + filename)
        self.batches = []
        self.words = 0
        self.num_hidden = 500
        self.num_visible = len(self.input[0])
        self.learning_rate = 0.01

        # for movie dataset
        self.title = data_movie.load_movie_title()
        self.tag = data_movie.load_movie_tag()
        self.overview = data_movie.load_movie_overview()
        self.movieId = data_movie.load_movie_ID()

        # Create Random generator
        self.numpy_rng = numpy.random.RandomState(1234)

        # Initial Weights and biases
        mu, sigma = 0, numpy.sqrt(0.01)
        self.weights = self.numpy_rng.normal(mu, sigma, (
            self.num_visible, self.num_hidden))

        # Inital hidden Bias
        self.hbias = numpy.zeros(self.num_hidden)

        # Inital visible Bias
        self.vbias = numpy.zeros(self.num_visible)

        self.delta_weights = numpy.zeros((self.num_visible, self.num_hidden))
        self.delta_hbias = numpy.zeros(self.num_hidden)
        self.delta_vbias = numpy.zeros(self.num_visible)

    # sigmoid function:
    def sigmoid(self, x):
        return 1. / (1 + numpy.exp(-x))

    # softmax function:
    def softmax(self, x):
        numerator = numpy.exp(x)
        denominator = numerator.sum(axis=1)
        denominator = denominator.reshape((x.shape[0], 1))
        softmax = numerator / denominator
        return softmax

    # Calculate and return Negative hidden states and probs
    def negativeProb(self, vis, hid, D):
        neg_vis = numpy.dot(hid, self.weights.T) + self.vbias
        softmax_value = self.softmax(neg_vis)
        neg_vis *= 0
        for i in xrange(len(vis)):
            neg_vis[i] = self.numpy_rng.multinomial(D[i], softmax_value[i], size=1)
        D = numpy.sum(neg_vis, axis=1)

        perplexity = numpy.nansum(vis * numpy.log(softmax_value))

        neg_hid_prob = self.sigmoid(numpy.dot(neg_vis, self.weights) + numpy.outer(D, self.hbias))
        return neg_vis, neg_hid_prob, D, perplexity

    # Train RSM model
    def train(self, max_epochs=15, batch_size=10, step=1, weight_cost=0.0002, momentum=0.9):
        data = self.input
        num_of_train = len(data)
        current_batch = batch_size
        while (current_batch + batch_size < num_of_train):
            self.batches.append(current_batch)
            current_batch += batch_size
        self.batches.append(num_of_train)
        for epoch in range(max_epochs):
            # Divide in to minibatch
            total_batch = len(self.batches)
            start_batch = 0
            reconstruction_error = 0
            perplexity = 0

            # Loop for each batch
            for batch_index in range(total_batch):
                # Get the data for each batch
                pos_vis = data[start_batch:self.batches[batch_index]]
                batch_size = len(pos_vis)
                start_batch = self.batches[batch_index]
                D = numpy.sum(pos_vis, axis=1)
                if epoch == 0:
                    self.words += numpy.sum(
                        pos_vis)  # Calculate the number of words in order to calculate the perplexity.
                # Caculate positive probs and Expectation for Sigma(ViHj) data
                pos_hid_prob = self.sigmoid(numpy.dot(pos_vis, self.weights) + numpy.outer(D, self.hbias))

                # If probabilities are higher than randomly generated, the states are 1
                randoms = self.numpy_rng.rand(batch_size, self.num_hidden)
                pos_hidden_states = numpy.array(randoms < pos_hid_prob, dtype=int)

                neg_vis = pos_vis
                neg_hid_prob = pos_hidden_states
                # Calculate negative probs and Expecatation for Sigma(ViHj) recon with k = 1,....
                for i in range(step):
                    neg_vis, neg_hid_prob, D, p = self.negativeProb(neg_vis, pos_hid_prob, D)
                    if i == 0:
                        perplexity += p

                # Update weight
                pos_products = numpy.dot(pos_vis.T, pos_hid_prob)
                pos_visible_bias_activation = numpy.sum(pos_vis, axis=0)
                pos_hidden_bias_activation = numpy.sum(pos_hid_prob, axis=0)
                neg_products = numpy.dot(neg_vis.T, neg_hid_prob)
                neg_visibe_bias_activation = numpy.sum(neg_vis, axis=0)
                neg_hidden_bias_activation = numpy.sum(neg_hid_prob, axis=0)

                # Update the weights and biases
                self.delta_weights = momentum * self.delta_weights + self.learning_rate * (
                    (pos_products - neg_products) / batch_size - weight_cost * self.weights)
                self.delta_vbias = (momentum * self.delta_vbias + (
                    pos_visible_bias_activation - neg_visibe_bias_activation)) * (self.learning_rate / batch_size)
                self.delta_hbias = (momentum * self.delta_hbias + (
                    pos_hidden_bias_activation - neg_hidden_bias_activation)) * (self.learning_rate / batch_size)
                self.weights += self.delta_weights
                self.vbias += self.delta_vbias
                self.hbias += self.delta_hbias

                reconstruction_error += numpy.square(pos_vis - neg_vis).sum()
            perplexity = numpy.exp(-perplexity / self.words)
            print('Epoch: {}, Error={}, Perplexity={}'.format(epoch, reconstruction_error, perplexity))

    # Get matrix data
    def preprocess_txt_rawdata(self, data_path):
        input = data_movie.get_bag_words_matirx(data_path)
        return input

    # Return top recommend list
    def recommendByTraindata(self, id, Rank=1):
        filename = OUTPUT_DATA_PATH + "output_data.dat"
        tmpHiddens = numpy.load(filename)["output"]
        testHidden = tmpHiddens[id]
        distance = []
        for tmpHidden in tmpHiddens:
            distance.append(spatial.distance.euclidean(testHidden, tmpHidden))
        distance[id] = numpy.inf
        ind = numpy.argsort(distance)[:Rank]
        return ind

    def updateOutput(self, new_input):
        filename = OUTPUT_DATA_PATH + "output_data.dat"
        tmpHiddens = numpy.load(filename)["output"]
        newHidden = self.getHiddenPro(new_input)
        tmpHiddens = numpy.concatenate([tmpHiddens, newHidden])
        with open(filename, "wb") as file:
            numpy.savez(file=file, output=tmpHiddens)

    def getHiddenPro(self, visible):
        output_batch = []
        for d in visible:
            out = self.sigmoid(numpy.dot(d, self.weights) + numpy.outer(sum(d), self.hbias))
            output_batch.append(out)
        return numpy.array(output_batch)

    def saveRsmWeights(self):
        filename = PARAMS_DATA_PATH + str(self.num_visible) + "_" + str(self.num_hidden) + "_weights_added_biases.dat"
        with open(filename, "wb") as file:
            numpy.savez(file=file, weights=self.weights, hbias=self.hbias, vbias=self.vbias)

    def saveTrainOutput(self, data):
        output = self.getHiddenPro(data)
        filename = OUTPUT_DATA_PATH + "output_data.dat"
        with open(filename, "wb") as file:
            numpy.savez(file=file, output=output)

    def loadTrainOutput(self):
        filename = OUTPUT_DATA_PATH + "output_data.dat"
        output = numpy.load(open(filename, "rb"))["output"]
        return output

    def retrain(self):
        result = {"key": "value"}
        data = self.input
        max_epochs = 60
        batch_size = 100
        step = 1
        self.train(max_epochs=max_epochs, batch_size=batch_size, step=step)
        self.saveTrainOutput(data)
        rank = 5
        txts = {}  # store movies' info
        for idx in range(len(data)):  # id indexed from 0
            txt = []
            recommend_list = self.recommendByTraindata(idx, Rank=rank)
            tmpList = []
            for tmprec in recommend_list:
                if (tmprec < 501):
                    tmpList.append(tmprec)
            recommend_list = tmpList
            txt.append("Movie ID " + str(self.movieId[idx]) + ", Title: " + str(self.title[idx]) + ", Tag: " + str(
                self.tag[idx]))
            txt.append("Overview: " + self.overview[idx] + "\n")
            txt.append("Recommend Movies List IDs: " + str([self.movieId[i] for i in recommend_list]))
            for j, m in enumerate(recommend_list):
                txt.append(
                    "Recommend movie top" + str(j + 1) + " ID " + str(self.movieId[m]) + ", Title: " + str(
                        self.title[m]) + ", Tag: " + str(self.tag[m]))
                txt.append("Overview: " + self.overview[m] + "\n")
            txts[idx] = txt
            recommend_title = ["(" + self.title[t] + ")" for t in recommend_list]
            result[self.movieId[idx]] = "::".join([str(self.movieId[i]) + j for i, j in zip(recommend_list, recommend_title)])
        filename = OUTPUT_DATA_PATH + "result_data.dat"
        # store the result
        with open(filename, "wb") as file:
            numpy.savez(file=file, output=result)
        for key in txts:
            txt_filename = OUTPUT_DATA_PATH + "movie_result_" + str(self.movieId[key]) + ".txt"
            txtfile = open(txt_filename, "wb")
            for item in txts[key]:
                txtfile.write("%s\n" % item)
        pickle.dump(result, open(os.path.dirname(__file__)+'/../data/topicRBM/output/Result.dat','wb'))
        return result

    #doc structure: title::tag::overview ex. Hello::drama|action::How are you?
    def update(self, new_doc_path, new_doc_id):
        result = {"key": "value"}
        # need also reload and store new doc id in storage
        new_input, data, self.movieId, self.title, self.tag, self.overview = data_movie.add_new_bag_words_matirx(INPUT_DATA_PATH+new_doc_path, new_doc_id)
        self.updateOutput(new_input)
        rank = 5
        txts = {}
        for idx in range(len(data)):
            txt = []
            recommend_list = self.recommendByTraindata(idx, Rank=rank)
            tmpList = []
            for tmprec in recommend_list:
                if (tmprec < 501):
                    tmpList.append(tmprec)
            recommend_list = tmpList
            txt.append(
                "Movie ID " + str(self.movieId[idx]) + ", Title: " + str(self.title[idx]) + ", Tag: " + str(
                    self.tag[idx]))
            txt.append("Overview: " + self.overview[idx] + "\n")
            txt.append("Recommend Movies List IDs: " + str([self.movieId[i] for i in recommend_list]))
            for j, m in enumerate(recommend_list):
                txt.append(
                    "Recommend movie top" + str(j + 1) + ", ID " + str(self.movieId[m]) + ", Title: " + str(
                        self.title[m]) + ", Tag: " + str(
                        self.tag[m]))
                txt.append("Overview: " + self.overview[m] + "\n")
            txts[idx] = txt
            recommend_title = ["(" + self.title[t] + ")" for t in recommend_list]
            result[self.movieId[idx]] = "::".join([str(self.movieId[i]) + j for i, j in zip(recommend_list, recommend_title)])
        filename = OUTPUT_DATA_PATH + "result_data.dat"
        # store the result
        with open(filename, "wb") as file:
            numpy.savez(file=file, result=result)
        for key in txts:
            txt_filename = OUTPUT_DATA_PATH + "movie_result_" + str(self.movieId[key]) + ".txt"
            txtfile = open(txt_filename, "wb")
            for item in txts[key]:
                txtfile.write("%s\n" % item)
        pickle.dump(result, open(os.path.dirname(__file__)+'/../data/topicRBM/output/Result.dat','wb'))
        return result


if __name__ == '__main__':
    filename = "OMDB_dataset_without_stopword.txt"
    newdoc = "777.txt"
    rsm = RSM(filename)
    result=rsm.retrain()
    print(len(result))
    result=rsm.update(newdoc, 777)
    print(len(result))
