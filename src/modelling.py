import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from catboost import CatBoostClassifier
import time

def model_based(X_train, y_train, X_test, y_test):
    
    # List of models
    models = [
        ('Random Forest', RandomForestClassifier()),
        ('XGBoost', XGBClassifier()),
        ('CatBoost', CatBoostClassifier(logging_level='Silent'))
    ]
    
    # Dataframe to store results
    results = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Training Time"])
    
    for name, model in models:
        start = time.time()  # Start time
        model.fit(X_train, y_train)
        end = time.time()  # End time
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        training_time = end - start
        
        # Append results to DataFrame
        results = results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Training Time": training_time
        }, ignore_index=True)
        
        # Print classification report
        print(name)
        print('Classification Report: \n', classification_report(y_test, y_pred))
    
    # Display results
    print(results)
    
    return results

# Example usage
# model_based(X_train, y_train, X_test, y_test)
