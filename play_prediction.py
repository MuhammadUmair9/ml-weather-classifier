
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Dataset
data = {
    'Day': ['Sunny', 'Windy', 'Sunny', 'Windy', 'Windy', 'Sunny', 'Windy', 'Sunny', 'Sunny', 'Windy', 'Sunny', 'Windy', 'Sunny', 'Sunny'],
    'Temprature': ['Cool', 'Cool', 'Hot', 'Hot', 'Hot', 'Cool', 'Hot', 'Hot', 'Hot', 'Hot', 'Cool', 'Hot', 'Hot', 'Hot'],
    'Class': ['Play', 'Not Play', 'Not Play', 'Play', 'Play', 'Play', 'Not Play', 'Play', 'Play', 'Play', 'Not Play', 'Not Play', 'Play', 'Play']
}

df = pd.DataFrame(data)
X_raw = df[['Day', 'Temprature']]
Y_raw = df['Class']

# Encoding
onehot_encoder = OneHotEncoder()
X_encoded = onehot_encoder.fit_transform(X_raw).toarray()

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_raw)

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y_encoded, test_size=0.3, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Neural Network (MLP)": MLPClassifier(max_iter=1000)
}

# Training and Evaluation
for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")

# Prediction for new instance
new_instance = pd.DataFrame([['Windy', 'Cool']], columns=['Day', 'Temprature'])
new_instance_encoded = onehot_encoder.transform(new_instance).toarray()

best_model = RandomForestClassifier()
best_model.fit(X_train, Y_train)
predicted_class = best_model.predict(new_instance_encoded)
predicted_label = label_encoder.inverse_transform(predicted_class)[0]

print("Prediction for Day=Windy, Temp=Cool:", predicted_label)
