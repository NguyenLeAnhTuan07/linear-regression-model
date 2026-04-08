# Linear Regression Model

## Model Overview
This project implements Linear Regression using **three approaches**:

### 1. Gradient Descent
The model parameters are optimized iteratively using **Gradient Descent**.
This method updates the weights step by step to minimize the Mean Squared Error (MSE) loss.
Gradient Descent is useful when:
- The dataset is large
- The number of features is high
- Closed-form solution is computationally expensive

---

### 2. Closed-form Solution (Normal Equation)
The model also supports the **closed-form solution**, where parameters are computed directly using the Normal Equation.
This approach is useful when:
- The dataset is small to medium
- A direct and exact solution is preferred
- No iterative optimization is needed

---

### 3. Stochastic Gradient Descent (SGD)
The model also implements Stochastic Gradient Descent (SGD), a variant of Gradient Descent where the model parameters are updated using only one training sample (or a small batch) at each iteration instead of the entire dataset.
Unlike standard Gradient Descent, which computes gradients over the full dataset, SGD updates weights more frequently, leading to faster convergence in many practical scenarios.
Stochastic Gradient Descent is useful when:
- The dataset is very large
- Faster updates are required
- Memory usage needs to be reduced
- Online learning (real-time data updates) is needed

Although SGD introduces more noise in the optimization process, it often converges faster and can escape shallow local minima more effectively.

---

## Dataset
The dataset used in this project is obtained from an external source.
You can replace the dataset to generate new weights and make predictions by using a dataset in CSV format.
Rename the dataset to `data.csv` and place it in the `data` directory.
Then, update the `feature_names.txt` file to match the feature names in the dataset.

> **Note:** The last feature name must be the output feature (the value you want to predict).

Additionally, note that there is an `encoding.py` file in the `encoding` directory.
If your dataset contains string values, you need to encode them into numerical values such as 0, 1, 2, 3, ... depending on how many unique categories the data contains.

🔗 Dataset links:  
- Car: https://www.kaggle.com/code/mahnazarjmand/car-data-prediction-with-linear-regression/input  
- Salary: https://www.kaggle.com/code/shubham47/linear-regression-salary-dataset/input

---

## Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Linux / macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install numpy pandas
```

---

## Run the Program

| Command | Description |
|---|---|
| `python train.py` | Train the model and save weights to `meta/wb.csv` |
| `python predict.py` | Make predictions using trained weights |
| `python evaluate.py` | Evaluate model performance |

---

## Evaluation (`evaluate.py`)
The evaluation script uses **K-Fold Cross Validation (k=5)** for all metrics.

**Part 1 — General Evaluation:**
- MSE and MAE are computed in scaled space then converted back to the original unit for interpretability
- R² Score is reported as-is (rescaling does not affect it)

**Part 2 — CV Score:**
- The user selects which metric to use as Mᵢ: MSE, MAE, or R²
- Reports `CV_Score_mean` and `CV_Score_std` across all folds

---

## Prediction (`predict.py`)
Place the data you want to predict in `predict/dudoan.csv`.

When `predict.py` runs, it will:
1. Encode and scale `dudoan.csv` using the parameters saved from training
2. Save the scaled input to `predict/dudoan_scaled.csv`
3. Run prediction and print results in the original unit

> **Note:** `dudoan.csv` must contain all input feature columns. Missing columns will raise an error listing which columns are absent.

---

## Project Structure
```
├── data/
│   ├── data.csv               # Raw dataset
│   ├── data_scaled.csv        # Scaled dataset (generated after train)
│   └── feature_names.txt      # List of feature names (last = target)
├── encoding/
│   └── encoding.py            # Categorical encoding
├── meta/
│   ├── scale_params.csv       # Mean/std saved from training
│   └── wb.csv                 # Trained weights and bias
├── modelpre/
│   ├── model.py               # fit() and predict() implementations
│   ├── preprocessing.py       # Encoding + scaling pipeline
│   └── scalestd.py            # Standard scaler utilities
├── predict/
│   ├── dudoan.csv             # Input data for prediction
│   └── dudoan_scaled.csv      # Scaled input (generated after predict)
├── evaluate.py
├── predict.py
└── train.py
```

---

## Conclusion
This project is a simple implementation of Linear Regression aimed at learning and understanding the core concepts behind the model.  
Users can easily modify the dataset, retrain the model, and make predictions with their own data.  
The project is intended for learning purposes and can be extended further for more advanced experiments.

Thank you for checking out this project.  
Have a great day! ☀️

**Author:** Nguyen Le Anh Tuan
