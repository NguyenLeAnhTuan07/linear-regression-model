## Model Overview

This project implements Linear Regression using **two approaches**:

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

### 3. Stochastic Gradient Descent (SGD)

The model also implements Stochastic Gradient Descent (SGD), a variant of Gradient Descent where the model parameters are updated using only one training sample (or a small batch) at each iteration instead of the entire dataset.

Unlike standard Gradient Descent, which computes gradients over the full dataset, SGD updates weights more frequently, leading to faster convergence in many practical scenarios.

Stochastic Gradient Descent is useful when:

- The dataset is very large

- Faster updates are required

- Memory usage needs to be reduced

- Online learning (real-time data updates) is needed

Although SGD introduces more noise in the optimization process, it often converges faster and can escape shallow local minima more effectively.
## Dataset

The dataset used in this project is obtained from an external source.
You can replace the dataset to generate new weights and make predictions by using a dataset in CSV format.
Rename the dataset to data.csv and place it in the data directory.
Then, update the feature_names.txt file to match the feature names in the dataset.
Note: the last feature name must be the output feature, which is the value you want to predict.

Additionally, note that there is an encoding.py file in the encoding directory.
If your dataset contains string values (for example, a car dataset), you need to encode them into numerical values such as 0, 1, 2, 3, ..., depending on how many unique categories the data contains.

🔗 Dataset link:  
car : https://www.kaggle.com/code/mahnazarjmand/car-data-prediction-with-linear-regression/input

salary: https://www.kaggle.com/code/shubham47/linear-regression-salary-dataset/input

## Create virtual environment (recommended)
python -m venv venv


Linux / macOS: source venv/bin/activate

Windows: venv\Scripts\activate


pip install numpy pandas

Run the program
Use this when you want to train the model and compute the weights: python train.py

Use this when you want to make predictions using the trained weights: python predict.py

## Conclusion

This project is a simple implementation of Linear Regression aimed at learning and understanding the core concepts behind the model.  
Users can easily modify the dataset, retrain the model, and make predictions with their own data.  
The project is intended for learning purposes and can be extended further for more advanced experiments.

Thank you for checking out this project.  
Have a great day! ☀️

**Author:** Nguyen Le Anh Tuan

