from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return request.form['json']