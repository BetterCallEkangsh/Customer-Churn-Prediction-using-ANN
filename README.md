# 🏦 Customer Churn Prediction using ANN

An end-to-end deep learning project that predicts whether a bank customer will **churn** (leave the bank) using an Artificial Neural Network built with **TensorFlow/Keras**, served through an interactive **Streamlit** web app.

The repo also includes a **hyperparameter tuning study** (GridSearchCV + SciKeras) and a **bonus ANN regression model** that estimates customer salary.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-green)

---

## 📌 Table of Contents

- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Preprocessing](#-preprocessing-pipeline)
- [Model Architecture](#-model-architecture-classification)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Bonus: Salary Regression](#-bonus-salary-regression)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Monitoring Training](#-monitoring-training)
- [Results](#-results)
- [Limitations & Future Work](#-limitations--future-work)
- [License](#-license)

---

## 🚀 Demo

The Streamlit app takes in customer details (geography, gender, age, balance, credit score, tenure, number of products, card/activity status, salary) and returns the **probability that the customer will churn**, along with a churn / no-churn verdict.

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
.
├── app.py                          # Streamlit web app for churn prediction
├── Churn_Modelling.csv             # Dataset
├── experiments.ipynb               # Preprocessing + ANN training (classification)
├── hyperparametertuningann.ipynb   # Hyperparameter tuning (GridSearchCV + KerasClassifier)
├── salaryregression.ipynb          # Bonus: ANN regression to predict EstimatedSalary
├── prediction.ipynb                # Load model & make single-customer predictions
├── model.h5                        # Trained ANN (classification)
├── label_encoder_gender.pkl        # Fitted LabelEncoder for 'Gender'
├── onehot_encoder_geo.pkl          # Fitted OneHotEncoder for 'Geography'
├── scaler.pkl                      # Fitted StandardScaler
└── requirements.txt                # Python dependencies
```

---

## 📊 Dataset

`Churn_Modelling.csv` contains **10,000 bank customer records**.

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | numeric | Customer credit score |
| `Geography` | categorical | France / Germany / Spain |
| `Gender` | categorical | Male / Female |
| `Age` | numeric | Customer age |
| `Tenure` | numeric | Years with the bank |
| `Balance` | numeric | Account balance |
| `NumOfProducts` | numeric | Bank products held |
| `HasCrCard` | binary | Has a credit card |
| `IsActiveMember` | binary | Active member flag |
| `EstimatedSalary` | numeric | Estimated annual salary |
| **`Exited`** | **target** | **1 = churned, 0 = retained** |

---

## 🧹 Preprocessing Pipeline

1. Drop identifier columns: `RowNumber`, `CustomerId`, `Surname`
2. **Label-encode** `Gender`
3. **One-hot encode** `Geography` → `Geography_France`, `Geography_Germany`, `Geography_Spain`
4. **Train/test split** with `random_state=42`
5. **Standardise** features with `StandardScaler` (fit on train only, applied to test)
6. Persist the fitted encoders and scaler as `.pkl` files so the app and notebooks apply the *identical* transformation at inference time

After preprocessing the model receives **12 input features**.

---

## 🧠 Model Architecture (Classification)

```
Input (12) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```

| Setting | Value |
|---|---|
| Total parameters | 2,945 |
| Optimizer | Adam (learning rate = 0.01) |
| Loss | Binary cross-entropy |
| Metric | Accuracy |
| Max epochs | 100 |
| Callbacks | `EarlyStopping` (monitor `val_loss`, patience 10, `restore_best_weights=True`) and `TensorBoard` |

---

## 🎛️ Hyperparameter Tuning

Choosing the number of hidden layers and neurons has no closed-form answer, so `hyperparametertuningann.ipynb` searches for it empirically instead of relying on guesswork.

### Approach

The ANN is wrapped in a **SciKeras `KerasClassifier`** so it behaves like a scikit-learn estimator and can be passed to **`GridSearchCV`**.

A model-builder function makes depth and width configurable:

```python
def create_model(neurons=32, layers=1):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))

    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(layers=1, neurons=32, build_fn=create_model, verbose=1)
```

### Search Space

| Hyperparameter | Values tried |
|---|---|
| `neurons` (per hidden layer) | 16, 32, 64, 128 |
| `layers` (hidden layers) | 1, 2 |
| `epochs` | 50, 100 |

```python
param_grid = {
    'neurons': [16, 32, 64, 128],
    'layers':  [1, 2],
    'epochs':  [50, 100]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    n_jobs=-1, cv=3, verbose=1)
grid_result = grid.fit(X_train, y_train)
```

- **4 × 2 × 2 = 16 candidate configurations**
- **3-fold cross-validation** → **48 model fits** in total
- Scoring: accuracy (default for classifiers)
- Preprocessing (encoding, scaling) is applied before the search; the scaler is fit on the training split only

### Best Result

| Metric | Value |
|---|---|
| **Best CV accuracy** | **0.8591** |
| **Best parameters** | `neurons = 16`, `layers = 2`, `epochs = 50` |

### Key Takeaways

- **Smaller beat bigger.** The winning network (2 hidden layers × 16 neurons) is *smaller* than the hand-built 64-32 baseline. For a tabular dataset of 10k rows and 12 features, extra capacity mostly adds overfitting risk rather than accuracy.
- **Fewer epochs won.** 50 epochs outperformed 100, which is consistent with the model starting to overfit when trained longer, and is why `EarlyStopping` is used in the main training notebook.
- **Accuracy plateaus around ~86%.** The baseline and tuned models land within a fraction of a percent of each other, which suggests the limit comes from the features and class imbalance rather than from architecture.

### Inspecting All 16 Results

```python
results = pd.DataFrame(grid_result.cv_results_)
print(results[['param_neurons', 'param_layers', 'param_epochs',
               'mean_test_score', 'std_test_score']]
      .sort_values('mean_test_score', ascending=False))
```

---

## 💰 Bonus: Salary Regression

`salaryregression.ipynb` reuses the same preprocessing pipeline to predict `EstimatedSalary` as a **regression** problem (`Exited` is kept as an input feature).

| Setting | Value |
|---|---|
| Architecture | Dense(64, ReLU) → Dense(32, ReLU) → Dense(1) *(linear output)* |
| Optimizer | Adam (learning rate = 0.01) |
| Loss / Metric | Mean Absolute Error |
| Callbacks | `EarlyStopping` (patience 20) + `TensorBoard` |
| Saved model | `regressionmodel.h5` |

The validation MAE settles at roughly **50,000**. Since `EstimatedSalary` in this dataset ranges from about 0 to 200k and is close to uniformly distributed, a model that always predicts the mean would score about the same. In other words, the available features carry very little signal about salary, so this notebook is best read as a demonstration of the regression workflow rather than a useful predictor.

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

- Python 3.10+
- tensorflow==2.15.0
- pandas, numpy
- scikit-learn
- scikeras
- matplotlib
- tensorboard
- streamlit

See `requirements.txt` for the full list.

---

## ▶️ Usage

### Run the Streamlit app
```bash
streamlit run app.py
```

### Retrain the model
Run `experiments.ipynb` end-to-end to reprocess the data, retrain the ANN, and regenerate `model.h5` plus the encoder/scaler `.pkl` files.

### Re-run hyperparameter tuning
Run `hyperparametertuningann.ipynb`. Note that the 48 fits can take a while on CPU.

### Predict for a single customer in a notebook
Use `prediction.ipynb`. Provide a dictionary of customer attributes:

```python
input_data = {
    'CreditScore': 600, 'Geography': 'France', 'Gender': 'Male',
    'Age': 40, 'Tenure': 3, 'Balance': 60000, 'NumOfProducts': 2,
    'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 50000
}
```

The notebook encodes and scales the input with the saved transformers, then outputs a churn probability. A probability **> 0.5** is classified as *likely to churn*.

---

## 📈 Monitoring Training

Training logs are written to `logs/fit/` (classification) and `log/fitreg/` (regression). Launch TensorBoard with:

```bash
tensorboard --logdir logs/fit
```

---

## ✅ Results

| Model | Task | Result |
|---|---|---|
| ANN 64-32 (main model) | Churn classification | ~86% validation accuracy |
| ANN tuned via GridSearchCV (2 × 16, 50 epochs) | Churn classification | 85.9% mean 3-fold CV accuracy |
| ANN 64-32 | Salary regression | Validation MAE ≈ 50k (no better than predicting the mean) |

---

## 🔭 Limitations & Future Work

- **Class imbalance:** only ~20% of customers churn, so accuracy flatters the model. Report precision, recall, F1 and ROC-AUC, and try class weights or SMOTE.
- **Tuned model not yet deployed:** `model.h5` is the hand-built 64-32 network. Retrain the best grid-search configuration with `EarlyStopping` and compare it on the held-out test set.
- **Wider search:** try `RandomizedSearchCV` or Keras Tuner / Optuna over learning rate, batch size, dropout, L2 regularisation and activation functions.
- **Modern Keras APIs:** replace `build_fn=` with `model=` in SciKeras, use an `Input(shape=...)` layer, and save models in the `.keras` format instead of legacy `.h5`.
- **Explainability:** add SHAP or permutation importance to show what drives churn (age, number of products and activity status are likely candidates).

---

## 📝 License

This project is open source and available for personal or educational use. Add a `LICENSE` file (e.g. MIT) if you plan to distribute it.