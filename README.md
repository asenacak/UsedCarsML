# Used Cars Price Prediction
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ADD8E6?style=for-the-badge&logo=XGBoost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)


This project involves predicting the prices of used cars using various machine learning models. The dataset used for this analysis comes from the [US Used Cars dataset (3 million)](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset), containing detailed information about used cars in the United States.

## Project Overview

The goal of this project is to build and compare multiple regression models to predict the prices of used cars. The models included in this analysis are:

- **Linear Regression**
- **Decision Trees**
- **Random Forest**
- **eXtreme Gradient Boosting (XGBoost)**
- **Deep Neural Networks (DNN)**

The performance of each model is evaluated using **Root Mean Squared Error (RMSE)** as the key metric. Additionally, techniques such as **GridSearchCV** and **Hyperband (Keras Tuner)** are used to fine-tune model parameters and achieve optimal performance.

## Data Source

The dataset used in this project is publicly available on Kaggle:

- **[US Used Cars dataset (3 million)](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)**

## Data Cleaning and Preprocessing

1. **Handling Placeholder Characters**: The dataset contains placeholder characters represented as `--`, which are cleaned before further processing.
   
2. **Missing Values**: A comprehensive approach is used to deal with missing data:
   - Removal of missing entries in certain columns.
   - Mean imputation, mode imputation, and multiple imputation techniques are applied where appropriate.

3. **Categorical Encoding**: One-hot encoding is applied to convert categorical variables into a numerical format that can be used by machine learning models.

4. **Normalization**: Numerical columns are standardized to ensure consistent scaling across features.

5. **Feature Selection**: The **SelectKBest** method is used to identify the most informative features for predictive modeling, enhancing model performance.

## Model Evaluation

After preprocessing, five regression models are trained and evaluated using **RMSE** as the key metric. The results are as follows:

- **Random Forest**: Best performance with an RMSE of **0.00265**.
- **XGBoost**: Second-best performance with an RMSE of **0.00278**.
- **Decision Tree**: RMSE of **0.00282**, closely following XGBoost.
- **Deep Neural Network (DNN)**: RMSE of **0.00283**, slightly higher than Decision Tree.
- **Linear Regression**: RMSE of **0.00393**, significantly worse than the other models.

## Model Tuning

- **GridSearchCV**: Used for hyperparameter tuning of the XGBoost, and Decision Tree models.
- **Keras Tuner (Hyperband)**: Applied for optimizing the Deep Neural Network model's parameters.

## Conclusion

Among the five models tested, **Random Forest** outperformed the others, achieving the lowest RMSE, indicating its superior ability to predict used car prices. **XGBoost**, **Decision Tree**, and **DNN** performed similarly, while **Linear Regression** lagged behind in terms of predictive accuracy.

## How to Run the Project

1. Clone this repository.
   ```bash
   git clone https://github.com/asenacak/UsedCarsML.git
   ```
2. Download the dataset from the following link: [US Used Cars dataset (3 million)](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset).
3. Install the necessary dependencies.
4. Run the Jupyter notebooks:

   * Used_Cars_Data_Cleaning.ipynb: Data preprocessing and cleaning.
   * models_usedcars.ipynb: Model building, evaluation, and hyperparameter tuning.

## Dependencies

* Python 3
* Jupyter Notebook
* pandas
* scikit-learn
* XGBoost
* TensorFlow/Keras
* Keras Tuner
* matplotlib

## License

This project is licensed under the MIT License.
