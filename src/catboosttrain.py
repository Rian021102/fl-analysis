import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import pickle
#import plot_roc_curve
from sklearn.metrics import roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
#import confusion matrix
from sklearn.metrics import confusion_matrix

def train_my_cat(X_train, y_train, X_test, y_test):
    # Create the model
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, 
                               loss_function='MultiClass')
    # Fit the model
    model.fit(X_train, y_train)
    # Save the model
    pickle.dump(model, open('model.pkl', 'wb'))
    # Predict the model
    y_pred = model.predict(X_test)
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # Calculate the precision
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f'Precision: {precision}')
    # Calculate the recall
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'Recall: {recall}')
    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1: {f1}')
    # Classification report
    print(classification_report(y_test, y_pred))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.show()
    return model