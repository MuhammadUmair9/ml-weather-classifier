# 🌤️ Play Prediction Based on Weather (ML Classifier Project)

This project is a simple machine learning classifier that predicts whether to "Play" or "Not Play" based on weather conditions such as "Day" (Sunny/Windy) and "Temperature" (Hot/Cool). The dataset is small and ideal for demonstrating classification models and feature encoding.

---

## 📊 Dataset

The dataset has three features:

| Day   | Temperature | Class     |
|-------|-------------|-----------|
| Sunny | Cool        | Play      |
| Windy | Cool        | Not Play  |
| ...   | ...         | ...       |

---

## 🔧 ML Workflow

1. **Data Preparation** (Pandas)
2. **Encoding**:
   - OneHotEncoder for input features
   - LabelEncoder for target variable
3. **Train-Test Split** (70% training, 30% testing)
4. **Model Training** using different ML models
5. **Prediction** and **Accuracy Evaluation**
6. **Prediction on New Sample**: `Day=Windy`, `Temp=Cool`

---

## 🤖 Models Used

The following 10 models are compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Gradient Boosting
8. AdaBoost
9. XGBoost (if available)
10. Neural Network (MLPClassifier)

---

## 📈 Output Example

```bash
Test Accuracy: 0.75
Prediction for Day=Windy, Temp=Cool: Not Play
