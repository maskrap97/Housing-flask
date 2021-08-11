import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lasso.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('house.html')

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = np.expm1(model.predict(final_features))

    output = round(prediction[0], 2)

    return render_template('house.html', prediction_text='Price of House Should be $ {}'.format(output))

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = np.expm1(model.predict([np.array(list(data.values()))]))

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)