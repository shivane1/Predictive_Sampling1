# Predictive_Sampling1

# Credit Card Fraud Analysis
This project implements fraud detection for credit card transactions using machine learning

# Steps

# Step 1: Import Libraries
```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

# Step 2: Read Transaction Data
```
transaction_data = pd.read_csv('Creditcard_data.csv')
```

# Step 3: Analyze Dataset Features
```
transaction_data.head()
transaction_data.info()
transaction_data.describe()
```
transaction_data.head(): Shows initial rows of transaction data
transaction_data.info(): Displays dataset structure and information
transaction_data.describe(): Generates statistical summary of numerical columns

# Step 4: Check Target Distribution
```
fraud_distribution = transaction_data["Class"].value_counts()
print("Transaction Types:")
print(fraud_distribution)
```
Analyzes distribution of fraud vs non-fraud transactions in dataset

# Step 5: Data Quality Check
```
null_count = transaction_data.isna().sum()
print("Null Values Per Feature:")
print(null_count)
```
Identifies missing or null values across all features

# Step 6: Split Transaction Types
```
normal_trans = transaction_data[transaction_data['Class'] == 0]
fraud_trans = transaction_data[transaction_data['Class'] == 1]
print('Normal transactions:', normal_trans.shape)
print('Fraudulent transactions:', fraud_trans.shape)
```
Separates dataset into fraudulent and normal transaction groups

# Step 7: Create Distribution Plot
```
plt.figure(figsize=(10, 5))
fraud_distribution.plot(kind='barh', color='lightblue', 
                       title="Transaction Type Distribution")
plt.xlabel("Count")
plt.ylabel("Class")
```
Visualizes transaction class distribution using horizontal bar plot

# Step 8: Balance Dataset
```
from imblearn.over_sampling import SMOTE
from collections import Counter

target = transaction_data['Class']
features = transaction_data.drop(['Class'], axis=1)

smote_balancer = SMOTE(random_state=42)
features_balanced, target_balanced = smote_balancer.fit_resample(features, target)
```
Uses SMOTE technique to create balanced dataset with equal class distribution

# Step 9: Combine Balanced Data
```
processed_data = pd.concat([
    pd.DataFrame(features_balanced),
    pd.DataFrame(target_balanced, columns=['Class'])
], axis=1)

print("Balanced dataset shape:", processed_data.shape)
print("Class distribution:\n", processed_data['Class'].value_counts())
```

# Step 10: Generate Sample Sets
```
from sklearn.model_selection import train_test_split

# 1. Random Sample
sample1 = processed_data.sample(n=int(0.2 * len(processed_data)), 
                              random_state=42)

# 2. Stratified Sample
grouped = processed_data.groupby('Class')
sample2 = grouped.apply(
    lambda x: x.sample(int(0.2 * len(x)), random_state=42)
).reset_index(drop=True)

# 3. Systematic Sample
interval = len(processed_data) // int(0.2 * len(processed_data))
offset = np.random.randint(0, interval)
sample3 = processed_data.iloc[offset::interval]

# 4. Cluster Sample
n_groups = 5
group_ids = np.arange(len(processed_data)) % n_groups
processed_data['Group'] = group_ids
selected_group = np.random.randint(0, n_groups)
sample4 = processed_data[processed_data['Group'] == selected_group].drop('Group', axis=1)

# 5. Bootstrap Sample
sample5 = processed_data.sample(n=int(0.2 * len(processed_data)), 
                              replace=True, 
                              random_state=42)

print("Sample sizes:", len(sample1), len(sample2), len(sample3), len(sample4), len(sample5))
```

# Step 11: Load ML Libraries
```
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

# Step 12: Initialize Models
```
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, 
                                                   learning_rate=0.1,
                                                   max_depth=3,
                                                   random_state=42),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier()
}
```

# Step 13: Prepare Results Storage
```
performance_metrics = {}
sample_set = [sample1, sample2, sample3, sample4, sample5]
```

# Step 14: Evaluate Model Performance
```
for model_name, classifier in classifiers.items():
    performance_metrics[model_name] = []
    
    for i, sample in enumerate(sample_set):
        X = sample.drop('Class', axis=1)
        y = sample['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        performance_metrics[model_name].append(accuracy)

results_table = pd.DataFrame(
    performance_metrics, 
    index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"]
)
print(results_table)
results_table.to_csv("model_accuracy.csv")
```

# Step 15: Save Final Results
```
results_table = pd.DataFrame(
    performance_metrics, 
    index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"]
)
print(results_table)
results_table.to_csv('Submission_102203194_kaustubh.csv')
```

# Best Model for each sample 
# Sample 1: Gradient Boosting 
# Sample 2: Logistic Regression
# Sample 3: Decision Tree / Gradient Boosting 
# Sample 4: Logistic Regression
# Sample 5: Decision Tree / Gradient Boosting 
