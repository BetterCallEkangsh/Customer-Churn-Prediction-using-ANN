# Customer Churn Prediction using ANN

A machine learning project that predicts whether a bank customer will churn (leave the bank) using an Artificial Neural Network (ANN) built with TensorFlow/Keras, deployed as an interactive Streamlit web app. The repo also includes a hyperparameter tuning experiment and a bonus regression model for estimating customer salary.

## 🚀 Demo

The app takes in customer details (geography, gender, age, balance, credit score, etc.) and predicts the probability that the customer will churn.

## 📂 Project Structure

```
.
├── app.py                          # Streamlit web app for churn prediction
├── Churn_Modelling.csv             # Dataset
├── experiments.ipynb               # Data preprocessing + ANN training (classification)
├── hyperparametertuningann.ipynb   # Hyperparameter tuning with GridSearchCV + KerasClassifier
├── salaryregression.ipynb          # Bonus: ANN regression model to predict EstimatedSalary
├── prediction.ipynb                # Notebook to load model & make single predictions
├── model.h5                        # Trained ANN model (classification)
├── label_encoder_gender.pkl        # Fitted LabelEncoder for 'Gender'
├── onehot_encoder_geo.pkl          # Fitted OneHotEncoder for 'Geography'
├── scaler.pkl                      # Fitted StandardScaler
└── requirements.txt                # Python dependencies
```

## 🧠 Model Overview

The dataset (`Churn_Modelling.csv`) contains bank customer records with features such as credit score, geography, gender, age, tenure, balance, number of products, credit card status, and activity status. The target variable is `Exited` (1 = churned, 0 = retained).

**Preprocessing steps:**
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
- Label-encoded `Gender`
- One-hot encoded `Geography`
- Scaled features with `StandardScaler`
- Train/test split (80/20)

**ANN architecture (classification):**
```
Input → Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)
```
Compiled with Adam optimizer and binary cross-entropy loss, trained with `EarlyStopping` and `TensorBoard` callbacks.

A second notebook (`salaryregression.ipynb`) reuses the same pipeline to instead predict `EstimatedSalary` as a regression problem (linear output layer, MAE metric).

Hyperparameter tuning (`hyperparametertuningann.ipynb`) uses `KerasClassifier` wrapped in `GridSearchCV` to search over number of neurons, layers, and epochs.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

### Run the Streamlit app
```bash
streamlit run app.py
```
This launches a browser-based UI where you can input customer details and get an instant churn prediction.

### Retrain the model
Open and run `experiments.ipynb` end-to-end to reprocess the data, retrain the ANN, and regenerate `model.h5` and the encoder/scaler pickle files.

### Run predictions in a notebook
Use `prediction.ipynb` to load the saved model/encoders/scaler and predict churn for a custom input record.

## 📦 Requirements

- Python 3.10+
- tensorflow==2.15.0
- pandas
- numpy
- scikit-learn
- scikeras
- matplotlib
- tensorboard
- streamlit

(see `requirements.txt` for the full list)

## 📊 Monitoring Training

Training logs are written to `logs/fit/` (classification) and `regressionlogs/fit/` (regression). View them with:
```bash
tensorboard --logdir logs/fit
```

## 📝 License

This project is open source and available for personal or educational use. Add a license file if you plan to distribute it.