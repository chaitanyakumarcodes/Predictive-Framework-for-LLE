from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from keras.models import load_model
import joblib

app = Flask(__name__)

model_counter = load_model('model_counter.h5')
model_cross = load_model('model_cross.h5')

scaler_counter = joblib.load('scaler_counter.save')
scaler_cross = joblib.load('scaler_cross.save')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/choose', methods=['POST'])
def choose():
    prediction_type = request.form.get('prediction_type')
    if prediction_type == 'counter':
        return redirect(url_for('counter'))
    elif prediction_type == 'cross':
        return redirect(url_for('cross'))
    else:
        return render_template('index.html', error="Invalid prediction type")

@app.route('/counter', methods=['GET', 'POST'])
def counter():
    if request.method == 'POST':
        try:
            input_alpha = request.form.get('alpha')

            if not input_alpha or not input_alpha.replace('.', '').isdigit():
                raise ValueError("Invalid input. Please enter a valid numerical value.")

            input_alpha = float(input_alpha)

            scaled_input_alpha = scaler_counter.transform(np.array([[input_alpha]]))

            predicted_value_counter = model_counter.predict(scaled_input_alpha)

            rounded_prediction_counter = int(round(predicted_value_counter[0, 0]))

            return render_template('counter.html', result_counter=rounded_prediction_counter, input_alpha=input_alpha)

        except ValueError as e:
            return render_template('counter.html', error=str(e))
    else:
        return render_template('counter.html')

@app.route('/cross', methods=['GET', 'POST'])
def cross():
    if request.method == 'POST':
        try:
            input_solvent = request.form.get('solvent')

            if not input_solvent or not input_solvent.replace('.', '').isdigit():
                raise ValueError("Invalid input. Please enter a valid numerical value.")

            input_solvent = float(input_solvent)

            scaled_input_solvent = scaler_cross.transform(np.array([[input_solvent]]))

            predicted_value_cross = model_cross.predict(scaled_input_solvent)

            rounded_prediction_cross = int(round(predicted_value_cross[0, 0]))

            return render_template('cross.html', result_cross=rounded_prediction_cross, input_solvent=input_solvent)

        except ValueError as e:
            return render_template('cross.html', error=str(e))
    else:
        return render_template('cross.html')

if __name__ == '__main__':
    app.run(debug=True)
