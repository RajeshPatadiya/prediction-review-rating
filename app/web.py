import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')


def binary_test(review):

    # 0 or 1, i.e, negative or positive
    return score

def multiple_test(review):

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
    happy = request.args.get('happy')
    term = request.args.get('srch-term')

    if happy == 'Y':
        score = binary_test(term)
    elif happy == 'N':
        score = multiple_test(term)
    else:
        score = "error"
    return render_template("result.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
