from flask import Flask
from src.exception import CustomException
import sys

try:
    from logger import logging
except ImportError:
    from src.logger import logging

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        raise Exception("This is a test exception for the index route.")
        
    except Exception as e:
        abc = CustomException(e, sys)
        logging.error("An error occurred in the index route.")
        return "Hello, Wkamran manzoor! Welcome to the ML Pipeline Project."

if __name__ == "__main__":
    app.run(debug=True)