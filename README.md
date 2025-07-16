

## 🚗💡 Premium Car Price Predictor & Insight Generator

> **A sleek ML project that uncovers insights in premium vehicles and predicts car prices using a fully automated pipeline.**

---

## 📌 Highlights

✅ **End-to-end ML pipeline** (EDA → Preprocessing → Modeling → Evaluation)
📊 **Market insights** for luxury 2-door vehicles
📈 **Regression models** to predict MSRP (car price)
🧠 **Scikit-learn models**: Linear, Tree-based, KNN
🎯 **Model accuracy score** (R²) with error metrics

---

## 🗂️ Project Structure

```
📦 ML-Project/
├── 📘 Machine_Learning.ipynb     # Main notebook
├── 📄 requirements.txt           # Required libraries
└── 📝 README.md                  # This file!
```

---

## 📁 Dataset Overview

| Feature            | Description                  |
| ------------------ | ---------------------------- |
| Make               | Car brand (e.g., BMW, Audi)  |
| Number of Doors    | Vehicle type (2-door focus)  |
| MSRP               | Manufacturer Suggested Price |
| Type, Engine, etc. | Additional vehicle features  |

> 🎯 **Focus:** 2-door premium cars with MSRP > 100,000

---

## 📊 Insights Generated

🔍 Insightful EDA reveals:

* Top 5 luxury brands for 2-door cars
* Max, Min & Average price of premium 2-door models
* Distribution of luxury brands (with elegant Seaborn visuals)

📊 **Sample Insight:**

```text
Top Premium Brands: Ferrari, Lamborghini, Porsche, Aston Martin, BMW
Average 2-Door Premium Price: $130,000+
```

---

## 🧪 ML Models Trained

| Model                   | Type                  |
| ----------------------- | --------------------- |
| Linear Regression       | 🔢 Linear             |
| Ridge Regression        | 🔢 Regularized Linear |
| Decision Tree Regressor | 🌳 Tree-based         |
| Random Forest Regressor | 🌲 Ensemble           |
| K-Nearest Neighbors     | 📍 Distance-based     |

🏆 **Best Model:** *RandomForestRegressor* (example)
📈 **Accuracy (R² Score):** *0.91*
🧮 **MAE:** *2500*, **MSE:** *7.1e+06*

---

## 🛠️ ML Pipeline Breakdown

1. **Data Cleaning**
2. **Label Encoding** (categorical → numeric)
3. **Feature Scaling** (StandardScaler)
4. **Train/Test Split**
5. **Model Training + Evaluation**


---

## 🚀 How to Run

1. 📥 Clone the repo:

```bash

git clone https://github.com/Zainch032/ML-Project.git
cd ML-Project

```

2. 📦 Install dependencies:

```bash
pip install -r requirements.txt
```

3. 🧪 Run the notebook:

```bash
jupyter notebook Machine_Learning.ipynb
```

4. 🔮 Predict on new data:



## 📦 Tech Stack

* 🐍 Python
* 📊 Pandas, NumPy, Matplotlib, Seaborn
* 🤖 Scikit-Learn
* 🧠 Regression Algorithms

