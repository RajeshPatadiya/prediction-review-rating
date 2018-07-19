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

app = Flask(__name__, static_url_path='/static')


def binary_test(review):
    score = 1
    # 0 or 1, i.e, negative or positive
    return score

def multiple_test(review):
    score = 1
    # 1, 2, 3, 4 or 5
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    #I don't know.. the path... pleas help me...
    tf.flags.DEFINE_string("checkpoint_dir", "./runs/1531944623/checkpoints", "Checkpoint directory from training run") 
    tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    
    s = review
    x_raw = [s]
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
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
            
            #I don't know.. to print..u...
            
            print("Predicts : ", all_predictions)
            print("Prediction Type", type(all_predictions[0]))

            score = all_predictions[0]
            
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
    binary_choice = request.args.get('happy')
    review = request.args.get('srch-term')

    if binary_choice == 'Y':
        score = binary_test(review)
    elif binary_choice == 'N':
        score = multiple_test(review)
    else:
        score = "error"
    return render_template("result.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
