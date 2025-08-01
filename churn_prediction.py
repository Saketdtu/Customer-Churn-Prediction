# Customer Churn Prediction Project
# Author: Saket Kumar, Delhi Technological University

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Customer Churn Prediction Analysis Started...")
print("=" * 50)

# Load the dataset
try:
    df = pd.read_csv('telco_churn.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Dataset shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: telco_churn.csv not found. Please download the dataset.")
    exit()

# Display basic information
print("\n📋 Dataset Info:")
print(df.head())
print(f"\n📈 Dataset columns: {list(df.columns)}")

# Data Cleaning and Preprocessing
print("\n🧹 Cleaning the data...")

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']

for column in categorical_columns:
    if column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Encode target variable
df['Churn'] = LabelEncoder().fit_transform(df['Churn'])

print(f"✅ Data cleaned! New shape: {df.shape}")

# Exploratory Data Analysis
print("\n📊 Creating visualizations...")

# Create output folder for plots
import os

os.makedirs('plots', exist_ok=True)

# 1. Churn Distribution
plt.figure(figsize=(10, 6))
churn_counts = df['Churn'].value_counts()
plt.subplot(1, 2, 1)
plt.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%')
plt.title('Churn Distribution')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='Churn')
plt.title('Churn Count')
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig('plots/churn_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Feature Analysis
plt.figure(figsize=(15, 10))

# Monthly Charges vs Churn
plt.subplot(2, 3, 1)
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges vs Churn')
plt.xticks([0, 1], ['No Churn', 'Churn'])

# Tenure vs Churn
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='Churn', y='tenure')
plt.title('Tenure vs Churn')
plt.xticks([0, 1], ['No Churn', 'Churn'])

# Contract vs Churn
plt.subplot(2, 3, 3)
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn.plot(kind='bar', ax=plt.gca())
plt.title('Contract Type vs Churn')
plt.legend(['No Churn', 'Churn'])
plt.xticks(rotation=45)

# Internet Service vs Churn
plt.subplot(2, 3, 4)
internet_churn = pd.crosstab(df['InternetService'], df['Churn'])
internet_churn.plot(kind='bar', ax=plt.gca())
plt.title('Internet Service vs Churn')
plt.legend(['No Churn', 'Churn'])
plt.xticks(rotation=45)

# Payment Method vs Churn
plt.subplot(2, 3, 5)
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'])
payment_churn.plot(kind='bar', ax=plt.gca())
plt.title('Payment Method vs Churn')
plt.legend(['No Churn', 'Churn'])
plt.xticks(rotation=45)

# Correlation heatmap
plt.subplot(2, 3, 6)
correlation_matrix = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation')

plt.tight_layout()
plt.savefig('plots/feature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Model Building
print("\n🤖 Building machine learning models...")

# Prepare features and target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Model 1: Logistic Regression
print("\n🔍 Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Model 2: Random Forest
print("🌳 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Results Comparison
print("\n📈 MODEL RESULTS:")
print("=" * 40)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy * 100:.2f}%)")
print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy * 100:.2f}%)")

# Choose best model
best_model = rf_model if rf_accuracy > lr_accuracy else lr_model
best_pred = rf_pred if rf_accuracy > lr_accuracy else lr_pred
best_name = "Random Forest" if rf_accuracy > lr_accuracy else "Logistic Regression"

print(f"\n🏆 Best Model: {best_name}")

# Detailed evaluation of best model
print(f"\n📊 Detailed Results for {best_name}:")
print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Feature Importance (if Random Forest)
if best_name == "Random Forest":
    plt.subplot(1, 2, 2)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')

plt.tight_layout()
plt.savefig('plots/model_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Business Insights
print("\n💡 BUSINESS INSIGHTS:")
print("=" * 40)

# Churn rate by key factors
insights = []

# Contract type analysis
contract_churn_rate = df.groupby('Contract')['Churn'].mean()
insights.append("📋 Contract Analysis:")
for contract, rate in contract_churn_rate.items():
    insights.append(f"   - Contract {contract}: {rate * 100:.1f}% churn rate")

# Tenure analysis
high_tenure = df[df['tenure'] > 24]['Churn'].mean()
low_tenure = df[df['tenure'] <= 12]['Churn'].mean()
insights.append(f"\n⏰ Tenure Analysis:")
insights.append(f"   - Low tenure (≤12 months): {low_tenure * 100:.1f}% churn rate")
insights.append(f"   - High tenure (>24 months): {high_tenure * 100:.1f}% churn rate")

# Monthly charges analysis
high_charges = df[df['MonthlyCharges'] > df['MonthlyCharges'].median()]['Churn'].mean()
low_charges = df[df['MonthlyCharges'] <= df['MonthlyCharges'].median()]['Churn'].mean()
insights.append(f"\n💰 Monthly Charges Analysis:")
insights.append(f"   - High charges: {high_charges * 100:.1f}% churn rate")
insights.append(f"   - Low charges: {low_charges * 100:.1f}% churn rate")

for insight in insights:
    print(insight)

print("\n🎯 RECOMMENDATIONS:")
print("=" * 40)
recommendations = [
    "1. Focus retention efforts on month-to-month contract customers",
    "2. Implement early intervention programs for customers with tenure < 12 months",
    "3. Review pricing strategy for high monthly charge customers",
    "4. Promote longer-term contracts with incentives",
    "5. Enhance customer support for electronic check payment users"
]

for rec in recommendations:
    print(rec)

print(f"""\n✅ Analysis complete! Model accuracy: {lr_accuracy * 100:.2f}%""")
print("📁 All plots saved to 'plots' folder")

