# importing necessary packages
from flask import Flask, render_template, request
import numpy as np
import pickle

# initializing the Flask application
app = Flask(__name__)

# loading the models
knn_model = pickle.load(
    open("./models/knn_model.pkl", "rb")
)

# loading the scaler
standard_scaler = pickle.load(
    open("./models/standard_scaler.pkl", "rb")
)

# loading the label dictionary
label_dict = pickle.load(
    open("./models/label_dict.pkl", "rb")
)

# adding routes to the application
@app.route("/", methods = ["GET", "POST"])
def home():

    # if the request method is POST then retrive the form values and do the prediction
    if request.method == "POST":
        try:
            
            column_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            form_values = request.form.to_dict()
            
            input_data = np.asarray(
                [float(form_values[i].strip()) for i in column_names]
                ).reshape(1, -1)
            
            # scaling the data
            input_data = standard_scaler.transform(input_data)

            # making prediction
            prediction = knn_model.predict(input_data)[0]
            
            # creating a message
            message = f"Growing {label_dict[prediction].upper()} would yield more crop"

            return render_template("home.html", prediction_message = message)
        
        except:
            message = "Please enter a valid input"
            return render_template("home.html", error_message = message)
        

    return render_template("home.html")

# running the app when the python file is executed
if __name__ == "__main__":
    app.run(debug=True)