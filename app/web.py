import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import load_model

import pickle

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

app = Flask(__name__, static_url_path='/static')

@app.before_first_request
def setting():
    # set tensorflow flags
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", './model_data', "Checkpoint directory from training run") 
    tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    global model
    model = load_model('./review.h5')
    model.summary()
    global graph
    graph = tf.get_default_graph()

def review_to_wordlist( review, remove_stopwords=True):
    # Data Pre-processing
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    #
    # 1. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 2. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 3. Optionally remove stop words (True by default)
    if remove_stopwords:
#        stops = set(stopwords.words("english"))
        stops = set(['those', 'ourselves', 'off', 'only', 'other', 'very', \
            're', 'over', 'should', 'mightn', 'then', 'where', 'some', 'aren', \
            'to', 'couldn', 'theirs', 'mustn', 'our', 'did', 'before', \
            'himself', 'her', 'needn', "aren't", 'him', 'she', 'he', 'through',\
            'shan', 'by', 'into', "didn't", 'myself', "should've", "wouldn't",\
            'if', 'they', 'same', 'wouldn', 'am', 'its', 'which', 'each',\
            'under', 'is', 'once', 'a', 'out', 'few', 'all', 'do', 'haven',\
            'an', 'yours', 'while', 'both', 've', 'what', 'ain', 'herself',\
            'themselves', 'for', 'will', 'have', "mustn't", 'more', 'that',\
            'ours', 'hers', 'doesn', 'no', 'your', 'just', 'below', 'll','isn',\
            'has', "don't", 'does', 'don', 'can', 'won', 'in', 'than', 'were',\
            'didn', "mightn't", "wasn't", 'd', 'against', 'most', 'been',\
            'during', "shan't", 'this', 'on', 'weren', 'hasn', 'up', 'be',\
            "it's", 'shouldn', 'these', 'so', 'because', 'm', "isn't",\
            "doesn't", 'here', 'and', 'we', 'between', 'itself', 'doing','you',\
            "haven't", 'his', 'after', 'as', 'until', 'own', "shouldn't",\
            'above', 'or', 'who', 'why', 'nor', 'it', 'again', 'yourselves',\
            'their', "that'll", "couldn't", "you're", 'how', 'my', 'down', 'o',\
            'me', 'further', 'whom', 'of', "you'll", "hasn't", 'being', 'i',\
            'now', 'at', 'the', "hadn't", 'wasn', 'with', 'ma', 'had',\
            "needn't", 'having', "you'd", 'there', "you've", 't',\
            'about', 'any', 'such', 'are', "won't", 'y', 'hadn', 'yourself',\
            's', 'was', 'but', 'too', "she's", "weren't", 'when', 'from', 'them'])
        words = [w for w in words if not w in stops]

    b=[]
    stemmer = english_stemmer #PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))

    # 4. Return a list of words
    return(b)

def binary_test(review):
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # fit "a review" to model input
    review_word_list = review_to_wordlist( review )
    sequences_test = tokenizer.texts_to_sequences([review])

#    print(review_word_list)
    '''
    print('sequences_test : ', sequences_test)
    new_seq = []
    for i in sequences_test:
        if i:
            new_seq.append(i[0])

    
    new_seq = [new_seq]
    '''
#    print('sequences_test : ', new_seq)
    X_review = sequence.pad_sequences(sequences_test, maxlen=30, value = 0)
#    X_review = sequence.pad_sequences(new_seq, maxlen=30, value = 0)
    print(X_review.shape)
    print('X_review : ', X_review)

    with graph.as_default():
        y_prob = model.predict(X_review)
        y_class = np.argmax(y_prob, axis=1)
        print('binary test\'s y_prob : ', y_prob)
        print('y_class : ', y_class[0])
        score = y_class[0]

    # 0 or 1, i.e, negative or positive
    return score

def multiple_test(review):
    FLAGS = tf.flags.FLAGS
    
    s = review
    x_raw = [s]
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
    print(vocab_path)
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            all_predictions = all_predictions.astype('int32')
            
            print("Predicts : ", all_predictions)
            print("Prediction Type", type(all_predictions[0]))

            score = all_predictions[0]
 
    # 1, 2, 3, 4 or 5
    return score

@app.route('/')
def homepage():
    title = "Epic Tutorials"
    try:
        return render_template("example_bootstrap.html")
    except Exception as e:
        return "Exception occur " + str(e)

@app.route('/hello')
def hello_world():
    return 'Hello World'

@app.route('/index')
def index_page():
    return render_template("index.html")

@app.route('/result')
def result_page():
    return render_template("result.html")

@app.route('/prediction', methods=['GET'])
def prediction():
    choice = request.args.get('method')
    review = request.args.get('srch-term')

    if choice == '0':
        score = binary_test(review)
    elif choice == '1':
        score = multiple_test(review)
    else:
        score = -1

    print('score : ', score)
    return render_template("result.html", score=score, choice=int(choice))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
