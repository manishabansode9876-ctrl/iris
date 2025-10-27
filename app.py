from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the trained model
with open("multiclass3.pkl", "rb") as f:   # change file name to your actual .pkl file
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Arrange data into correct shape
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get probabilities
        probabilities = model.predict_proba(input_data)[0]  # array like [0.1, 0.8, 0.1]
        class_labels = model.classes_  # e.g. ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

        # Combine labels with probabilities
        prob_text = ", ".join([f"{cls}: {round(prob*100,2)}%" 
                               for cls, prob in zip(class_labels, probabilities)])

        return render_template(
            'index.html',
            prediction_text=f"The predicted Iris flower is: {prediction}",
            probability_text=f"Class probabilities → {prob_text}"
        )

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)


     
     

