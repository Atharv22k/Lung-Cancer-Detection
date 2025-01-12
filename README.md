# Lung Cancer Detection

This repository contains a **Lung Cancer Detection System** that leverages machine learning to predict the likelihood of lung cancer based on survey data. The project includes a Flask-based web application and a pre-trained model for easy deployment and use.

---

## Features

- **Machine Learning Powered**: Predict lung cancer likelihood based on user inputs.
- **Flask Web App**: Interactive web interface for making predictions.
- **Pre-trained Model**: Comes with a trained model ready to use.
- **Customizable**: Allows for retraining with new datasets.
- **Real-time Predictions**: Input data and get instant results.
- **Yes/No Questions**: Predict lung cancer based on answers to yes/no questions.

---

## Prerequisites

Before running this project, ensure you have the following:

- Python 3.7 or later
- pip (Python package manager)
- A virtual environment (recommended for dependency isolation)

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Atharv22k/Lung-Cancer-Detection.git
   cd Lung-Cancer-Detection
   ```

2. **Set Up Virtual Environment** (Optional but Recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Required Files Are Present**:

   - `lung_cancer_survey.csv`: Dataset for training (if retraining is required).
   - `lung_cancer_model.joblib`: Pre-trained model for predictions.
   - `gender_encoder.joblib`: Preprocessing encoder for gender data.

---

## Usage

1. **Run the Flask Application**:

   ```bash
   python app.py
   ```

   By default, the application will run at `http://127.0.0.1:5000/`.

2. **Access the Web Interface**:

   Open a web browser and navigate to `http://127.0.0.1:5000/` to use the application.

3. **Make Predictions**:

   Enter survey details into the form and submit to receive predictions. The application uses yes/no answers to specific questions to determine the likelihood of lung cancer.

---

## Project Structure

```
Lung-Cancer-Detection/
├── app.py                # Main Flask application
├── lung_cancer_model.joblib # Pre-trained machine learning model
├── gender_encoder.joblib    # Gender encoder for preprocessing
├── lung_cancer_survey.csv   # Dataset for training
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m 'Add some feature'
   ```

4. Push to the branch:

   ```bash
   git push origin feature-name
   ```

5. Open a pull request.

---


## Acknowledgments

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Python Official Documentation](https://www.python.org/doc/)
