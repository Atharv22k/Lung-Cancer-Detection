from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('lung_cancer_model.joblib')

# Manual gender encoding
def encode_gender(gender):
    return 0 if gender == 'M' else 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Collect form data
    features = [
        encode_gender(request.form['gender']),
        int(request.form['age']),
        1 if request.form.get('smoking') == 'no' else 2,
        1 if request.form.get('yellow_fingers') == 'no' else 2,
        1 if request.form.get('anxiety') == 'no' else 2,
        1 if request.form.get('peer_pressure') == 'no' else 2,
        1 if request.form.get('chronic_disease') == 'no' else 2,
        1 if request.form.get('fatigue') == 'no' else 2,
        1 if request.form.get('allergy') == 'no' else 2,
        1 if request.form.get('wheezing') == 'no' else 2,
        1 if request.form.get('alcohol_consumption') == 'no' else 2,
        1 if request.form.get('coughing_of_blood') == 'no' else 2,
        1 if request.form.get('shortness_of_breath_in_calm') == 'no' else 2,
        1 if request.form.get('swallowing_difficulty') == 'no' else 2,
        1 if request.form.get('chest_pain') == 'no' else 2
    ]
    
    prediction = model.predict([features])
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
