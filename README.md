# 📦 Product Quantity Predictor 
--
check here <a href="https://huggingface.co/spaces/Krishna5488/Sales_QTY_prediction" target="blank_"></a>
This project predicts the **sales quantity of a product** based on the selected product name, price bucket, month, and year. It uses **Word2Vec** for product text embeddings and **RandomForest Regression** for quantity prediction. A user-friendly Streamlit interface allows interactive predictions.

---

## 🚀 Features

- 📦 Select from 200+ products
- 💰 Choose a price range bucket
- 📅 Select month and year
- 📈 Predict expected quantity with a click

---

## 🧠 ML Workflow

### 🔹 Word2Vec Embeddings
Product names are tokenized and passed through a pretrained **Word2Vec model** to generate semantic embeddings.

### 🔹 Features Used
- Product vector (Word2Vec mean)
- Year
- Month (as integer)
- Midpoint of price bucket

### 🔹 Models Evaluated
Several regression models were evaluated. The final model chosen was **XGBoost Regressor**, which outperformed others in generalization.

---

## 📊 Model Evaluation

| Model                    | Dataset | R² Score | MAE     | RMSE      |
| ------------------------ | ------- | -------- | ------- | --------- |
| **Linear Regression**    | Test    | -0.3120  | 21.6639 | 1027.6646 |
| **Support Vector Regr.** | Test    | 0.5465   | 9.2988  | 18.8463   |
| **Decision Tree Regr.**  | Test    | 0.7386   | 8.6600  | 14.3100   |
| **KNN Regressor**        | Test    | 0.1413   | 13.1600 | 25.9400   |
| **XGBoost Regressor**    | Test    | 0.8127   | 7.2389  | 146.7239  |
| **Random Forest Regr.**  | Test    | 0.8356   | 6.6459  | 128.7701  |


✅ **Best Performance**: RandomForest Regressor

---

## 🖥️ Tech Stack

- Python 🐍
- Streamlit 🚀
- Word2Vec (Gensim)
- XGBoost
- Scikit-learn
- NumPy, Joblib

