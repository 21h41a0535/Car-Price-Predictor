from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model with error handling
try:
    model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
except EOFError:
    raise Exception("The model file is empty or corrupted. Please check the file.")
except Exception as e:
    raise Exception(f"An error occurred while loading the model: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Initialize the scaler
scaler = StandardScaler()

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data
            year = int(request.form['Year'])
            present_price = float(request.form['Present_Price'])
            kms_driven = int(request.form['Kms_Driven'])
            kms_driven_log = np.log(kms_driven)
            owner = int(request.form['Owner'])

            # Fuel type encoding
            fuel_type_petrol = request.form['Fuel_Type_Petrol']
            fuel_type_diesel = 1 if fuel_type_petrol == 'Petrol' else 0

            # Calculate the age of the car
            year = 2020 - year

            # Seller type encoding
            seller_type_individual = 1 if request.form['Seller_Type_Individual'] == 'Individual' else 0

            # Transmission type encoding
            transmission_manual = 1 if request.form['Transmission_Mannual'] == 'Mannual' else 0

            # Prepare the input features for prediction
            features = np.array([[present_price, kms_driven_log, owner, year,
                                  fuel_type_diesel, fuel_type_petrol,
                                  seller_type_individual, transmission_manual]])

            # Make a prediction
            prediction = model.predict(features)
            output = round(prediction[0], 2)

            # Return the prediction result
            if output < 0:
                return render_template('index.html', prediction_text="Sorry, you cannot sell this car.")
            else:
                return render_template('index.html', prediction_text="You can sell the car at ₹ {}".format(output))

        except Exception as e:
            return render_template('index.html', prediction_text="Error occurred: {}".format(str(e)))

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
