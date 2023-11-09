from flask import Flask, render_template,request
# Load model directly
from transformers import pipeline

app = Flask(__name__)

pipe = pipeline("text-classification", \
    model="distilbert-base-uncased-finetuned-sst-2-english", \
    token="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")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    result = pipe(text)[0]

    return render_template('index.html',\
        text=text, result=result)

if __name__ == '__main__':
    app.run(debug=True)


