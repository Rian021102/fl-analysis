from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

PARAM_GRID = [
    {   
        #param grid for XGBoost
        'classifier__eta': [0.01, 0.1, 0.3, 0.5, 0.7, 1],
        'classifier__min_child_weight': [1, 5, 10],
        'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__gamma': [0, 0.25, 0.5, 1.0],
        'classifier__max_delta_step': [0, 0.25, 0.5, 1.0],
        'classifier__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'classifier__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'classifier__colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'classifier__reg_lambda': [0.01, 0.1, 1.0],
        'classifier__reg_alpha': [0, 0.1, 1.0]
    }
]

def train(X_train_sm, y_train_sm, X_test, y_test):
    # GridSearchCV
    pipe = Pipeline([('classifier', XGBClassifier())])
    clf = RandomizedSearchCV(pipe, PARAM_GRID, cv=5, verbose=0, n_jobs=4)
    best_clf = clf.fit(X_train_sm, y_train_sm)

    # Predict
    y_pred = best_clf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Change average value
    recall = recall_score(y_test, y_pred, average='weighted')  # Change average value
    f1 = f1_score(y_test, y_pred, average='weighted')  # Change average value
    classificationreport=classification_report(y_test, y_pred)

    # Print metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'Classification Report: {classificationreport}')


    # Save trained model
    with open('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/models/model.pkl', 'wb') as f:
        pickle.dump(best_clf, f)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'Classification Report': classificationreport

    }
