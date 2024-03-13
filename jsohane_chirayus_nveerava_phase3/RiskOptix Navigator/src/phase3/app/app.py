# importing necessary Libraries
from flask import Flask, render_template, request
import joblib
import numpy as np
import logging

# Creating an instance of flask application
app = Flask(__name__)

# Setting up logging to write logs to both a file(app.log) and console
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_instance = logging.StreamHandler()
console_instance.setLevel(logging.INFO)

# Adding the console handler to the Flask applications logger
app.logger.addHandler(console_instance)

# Loading the pickel file of pretrained XGBoost model
model = joblib.load("xgboost_model.pkl")

# Loading the pickle file of our normalization parameters
scaler = joblib.load("standard_scaler.pkl")

# Home route defined for rendering index.html from templates folder


@app.route('/')
def home():
    return render_template('index.html')

# Defining route for prediction page, which will handle form submission


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Extracting form values
        form_values = list(request.form.values())

        # excluding name column from processing
        features_test = [float(x) for x in form_values[1:]]
        sliced_f = features_test[0:9]
        remaining_f = features_test[9:15]
        arr = [0]*18
        conc = remaining_f + arr

        # Scaling back the features to model compatible form
        scaled_features = scaler.transform([sliced_f])

        # flattening back to a 1D array
        scaled_flattened = scaled_features.flatten()
        final_f = result = np.concatenate([scaled_flattened, conc])

        # Updating the occupation type value as per user selection
        final_f[(int)(features_test[-1]-1)] = 1

        # Reshaping input data and logging it for debugging purpose
        input_data = np.array(final_f).reshape(1, -1)
        app.logger.info(f'Input data: {input_data}')

        # Making prediction using loaded model
        prediction = model.predict(input_data)[0]

        # Rendering the result.html template with the prediction and users name
        return render_template('result.html', prediction=prediction, name=form_values[0],credit_score=form_values[9],default_in_last_6months=form_values[11],credit_limit=form_values[8])

    # If request method is not POST, rendering the predict.html template
    return render_template('predict.html')


@app.route('/index')
def go_to_index():
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
