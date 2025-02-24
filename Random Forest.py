import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('breast_cancer_data.csv')

# Encode the diagnosis column to numerical values
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Split the dataset into features (X) and target (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Dataset Splitting
# Holdout method: 80-20 split
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Holdout method: 70-30 split
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 2. Exploratory Data Analysis (EDA)
# Heatmap for correlation
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Pie chart for class distribution
class_distribution = df['diagnosis'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_distribution, labels=['Benign', 'Malignant'], autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'])
plt.title('Pie Chart of Class Distribution')
plt.show()

# Bar chart for class distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=class_distribution.index, y=class_distribution.values, palette='viridis')
plt.title('Bar Chart of Class Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()

# 3. Apply Random Forest and share results
def evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Performance measures
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")

    # ROC Curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Evaluate on 80-20 split
print("Results on 80-20 Split:")
evaluate_model(X_train_80, X_test_20, y_train_80, y_test_20)

# Evaluate on 70-30 split
print("\nResults on 70-30 Split:")
evaluate_model(X_train_70, X_test_30, y_train_70, y_test_30)
