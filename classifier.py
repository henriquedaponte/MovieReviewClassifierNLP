import scipy
from scipy import sparse
from scipy.sparse import csc_matrix
import numpy as np
from collections import Counter
import string

# utility functions
def load_data(file_name):
    '''
    @input:
     file_name: a string. should be either "training.txt" or "texting.txt"
    @return:
     a list of sentences
    '''
    with open(file_name, "r", encoding='utf-8') as file:
        sentences = file.readlines()
    return sentences

def tokenize(sentence):
    # Convert a sentence into a list of words
    wordlist = sentence.translate(str.maketrans('', '', string.punctuation)).lower().strip().split(
        ' ')

    return [word.strip() for word in wordlist]


# Main "Feature Extractor" class:
# It takes the provided tokenizer and vocab as an input.
class feature_extractor:
    def __init__(self, vocab, tokenizer):
        self.tokenize = tokenizer
        self.vocab = vocab  # This is a list of words in vocabulary
        self.vocab_dict = {item: i for i, item in
                           enumerate(vocab)}  # This constructs a word 2 index dictionary
        self.d = len(vocab)

    def bag_of_word_feature(self, sentence):
        '''
        Bag of word feature extactor
        :param sentence: A text string representing one "movie review"
        :return: The feature vector in the form of a "sparse.csc_array" with shape = (d,1)
        '''

        # Tokenize the sentence
        words = self.tokenize(sentence)

        # Count the word occurrences
        word_counts = Counter(words)

        # Filter out words not in the vocabulary
        filtered_counts = {self.vocab_dict[word]: count for word, count in word_counts.items() if word in self.vocab_dict}

        # Create the sparse csc_matrix
        row_indices = list(filtered_counts.keys())
        data = list(filtered_counts.values())
        col_indices = [0] * len(row_indices)

        x = sparse.csc_array((data, (row_indices, col_indices)), shape=(self.d, 1))

        return x


    def __call__(self, sentence):
        # This function makes this any instance of this python class a callable object
        return self.bag_of_word_feature(sentence)


class data_processor:
    '''
    Please do NOT modify this class.
    This class basically takes any FeatureExtractor class, and provide utility functions
    1. to process data in batches
    2. to save them to npy files
    3. to load data from npy files.
    '''
    # This class
    def __init__(self,feat_map):
        self.feat_map = feat_map

    def batch_feat_map(self, sentences):
        '''
        This function processes data according to your feat_map. Please do not modify.

        :param sentences:  A single text string or a list of text string
        :return: the resulting feature matrix in sparse.csc_array of shape d by m
        '''
        if isinstance(sentences, list):
            X = scipy.sparse.hstack([self.feat_map(sentence) for sentence in sentences])
        else:
            X = self.feat_map(sentences)
        return X

    def load_data_from_file(self, filename):
        # The filename should be *.npy
        with open(filename, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            if data.shape == ():
                X = data[()]
            else:
                X = data
        return X

    def process_data_and_save_as_file(self,sentences,labels, filename):
        # The filename should be *.npy
        X = self.batch_feat_map(sentences)
        y = np.array(labels)
        with open(filename, 'wb') as f:
            np.save(f, X, allow_pickle=True)
        return X, y


class classifier_agent():
    def __init__(self, feat_map, params):
        '''
        This is a constructor of the 'classifier_agent' class. Please do not modify.

         - 'feat_map'  is a function that takes the raw data sentence and convert it
         into a data vector compatible with numpy.array

         Once you implement Bag Of Word and TF-IDF, you can pass an instantiated object
          of these class into this classifier agent

         - 'params' is an numpy array that describes the parameters of the model.
          In a linear classifer, this is the coefficient vector. This can be a zero-initialization
          if the classifier is not trained, but you need to make sure that the dimension is correct.
        '''
        self.feat_map = feat_map
        self.data2feat = data_processor(feat_map)
        self.batch_feat_map = self.data2feat.batch_feat_map

        self.params = np.array(params)

    def score_function(self, X):
        '''
        This function computes the score function of the classifier.
        Note that the score function is linear in X
        :param X: A scipy.sparse.csc_array of size d by m, each column denotes one feature vector
        :return: A numpy.array of length m with the score computed for each data point
        '''

        (d,m) = X.shape
        d1= self.params.shape[0]
        if d != d1:
            self.params = np.array([0.0 for i in range(d)])

        s = np.zeros(shape=m)  # this is the desired type and shape for the output

        # Computing the score which is the dot product of the features (X) and weights (self.params)
        
        # Compute matrix multiplication of X^T and w (self.params)
        s = X.T.dot(self.params)

        return np.array(s)

    def predict(self, X, RAW_TEXT=False, RETURN_SCORE=False):
        '''
        This function makes a binary prediction or a numerical score
        :param X: d by m sparse (csc_array) matrix
        :param RAW_TEXT: if True, then X is a list of text string
        :param RETURN_SCORE: If True, then return the score directly
        :return:
        '''
        if RAW_TEXT:
            X = self.batch_feat_map(X)

        # Computing scores
        s = self.score_function(X)

        # If true returns score directly
        if(RETURN_SCORE):
            return s

        # Initializing predictions array
        preds = np.zeros(shape=X.shape[1])

        # Using vectorized operation (better performance) to make predictions (1 if review is positive, 0 if it is negative)
        preds = (s > 0).astype(int)

        return preds

    def error(self, X, y, RAW_TEXT=False):
        '''
        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :param RAW_TEXT: if True, then X is a list of text string,
                        and y is a list of true labels
        :return: The average error rate
        '''
        if RAW_TEXT:
            X = self.batch_feat_map(X)
            y = np.array(y)

        # Making predictions on data
        predict = self.predict(X)
 
        # Calculating error rate using vectorized operations (better performance)
        err =  np.mean(predict != y)

        return err

    def loss_function(self, X, y):
        '''
        This function implements the logistic loss at the current self.params

        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :return:  a scalar, which denotes the mean of the loss functions on the m data points.

        '''
        # Calculate probability of the positive loss using logistic function
        scores = np.array(self.score_function(X)).reshape(-1, 1)
        exponential = np.array(np.exp(scores)).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
       
        left = -scores.dot(y.T)

        diag = np.diag(left).reshape(-1, 1)
    
        # Calculating binary cross-entropy loss
        loss = diag + np.log(1 + exponential)

        # Return loss
        return np.mean(loss)
    
    def gradient(self, X, y):
        '''
        It returns the gradient of the (average) loss function at the current params.
        :param X: d by m sparse (csc_array) matrix
        :param y: m dimensional vector (numpy.array) of true labels
        :return: Return an nd.array of size the same as self.params
        '''

        # Initializing gradient matrix
        grad = np.zeros_like(self.params)
      
        # Computing scores = w^T @ X
        scores = self.score_function(X)
        
        # Reshaping y
        y = np.array(y).reshape(-1, 1)
        
        # Calculating exponetial = e^(X^T @ W)
        exponential = np.array(np.exp(scores)).reshape(-1, 1)
     
        # Getting number of samples to normalize result
        m = X.shape[1]

        # Calculating the gradient (partial derivative with respect to weigths from loss function)
        grad = (1/m) * (-X.dot(y) + (X.dot(exponential / (1 + exponential))))

        return grad.reshape(-1)

    def train_gd(self, train_sentences, train_labels, niter, lr=0.01, RAW_TEXT=True):
        '''
        The function should updates the parameters of the model for niter iterations using Gradient Descent
        It returns the sequence of loss functions and the sequence of training error for each iteration.

        By default the function takes raw text. But it also takes already pre-processed features,
        if RAW_TEXT is set to False.

        :param train_sentences: Training data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param train_labels: Training data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :param niter: number of iterations to train with Gradient Descent
        :param lr: Choice of learning rate (default to 0.01, but feel free to tweak it)
        :return: A list of loss values, and a list of training errors.
                (Both of them has length niter + 1)
        '''
        if RAW_TEXT:
            # the input is raw text
            Xtrain = self.batch_feat_map(train_sentences)
            ytrain = np.array(train_labels).reshape(-1, 1)
        else:
            # the input is the extracted feature vector
            Xtrain = train_sentences
            ytrain = train_labels


        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]

     
        # Solution:
        for i in range(niter):
            # TODO ======================== YOUR CODE HERE =====================================
            # You need to iteratively update self.params
            self.params = self.params - lr * self.gradient(Xtrain, ytrain)

            # TODO =============================================================================
            train_losses.append(self.loss_function(Xtrain, ytrain))
            train_errors.append(self.error(Xtrain, ytrain))

            if i % 100 == 0:
                print('iter =',i,'loss = ', train_losses[-1],
                  'error = ', train_errors[-1])

        return train_losses, train_errors

    def train_sgd(self, train_sentences, train_labels, nepoch, lr=0.001, RAW_TEXT=True):
        '''
        The function should updates the parameters of the model for using Stochastic Gradient Descent.
        (random sample in every iteration, without minibatches,
        pls follow the algorithm from the lecture which picks one data point at random).

        By default the function takes raw text. But it also takes already pre-processed features,
        if RAW_TEXT is set to False.


        :param train_sentences: Training data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param train_labels: Training data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :param nepoch: Number of effective data passes.  One data pass is the same as n iterations
        :param lr: Choice of learning rate (default to 0.001, but feel free to tweak it)
        :return: A list of loss values and a list of training errors.
                (initial loss / error plus  loss / error after every epoch, thus length epoch +1)
        '''


        if RAW_TEXT:
            # the input is raw text
            Xtrain = self.batch_feat_map(train_sentences)
            ytrain = np.array(train_labels)
        else:
            # the input is the extracted feature vector
            Xtrain = train_sentences
            ytrain = train_labels

        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]


        # First construct the dataset
        # then train the model using SGD
        # Solution
        sampler = 1/len(ytrain)
        niter = int(nepoch / sampler)
    
        #params = sparse.csr_array(self.params)

        for i in range(nepoch):
            for j in range(len(ytrain)):

                # Selecting the index of one random data point.
                idx = np.random.choice(len(ytrain), 1)

                # Extract the data point (feature vector and label) at the chosen index
                X_sample = Xtrain[:, idx]
                y_sample = ytrain[idx]

            # Update self.params using stochastic gradient descent
                self.params = self.params - lr * self.gradient(X_sample, y_sample)

            # logging
            train_losses.append(self.loss_function(Xtrain, ytrain))
            train_errors.append(self.error(Xtrain, ytrain))

            print('epoch =',i,'iter=',i*len(ytrain)+j+1,'loss = ', train_losses[-1],
                  'error = ', train_errors[-1])


        return train_losses, train_errors




    def eval_model(self, test_sentences, test_labels, RAW_TEXT=True):
        '''
        This function evaluates the classifier agent via new labeled examples.
        Do not edit please.
        :param test_sentences: Test data, a list of text strings;
            when "RAW_TEXT" is set to False, this input is a d by n numpy.array or scipy.csc_array
        :param test_labels: Test data, a list of labels 0 or 1
            when "RAW_TEXT" is set to False, this input is a n dimensional numpy.array
        :return: error rate on the input dataset
        '''

        if RAW_TEXT:
            # the input is raw text
            X = self.batch_feat_map(test_sentences)
            y = np.array(test_labels)
        else:
            # the input is the extracted feature vector
            X = test_sentences
            y = test_labels

        return self.error(X, y)

    def save_params_to_file(self, filename):
        # The filename should be *.npy
        with open(filename, 'wb') as f:
            np.save(f, self.params)

    def load_params_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.params = np.load(f)
