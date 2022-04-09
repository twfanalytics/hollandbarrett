import pdb
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, log_loss
from matplotlib import pyplot as plt


def calculate_metrics(clf, y_test, x_test):
    # Predict the response for test dataset
    y_proba = clf.predict_proba(x_test)
    y_pred = clf.predict(x_test)

    plt.hist(y_proba[:, 1])
    plt.show()

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    # F1-score, harmonic mean
    print("F1-score: ", f1_score(y_test, y_pred))

    # ROC and AUC, keep probabilities for the positive outcome only
    y_proba = y_proba[:, 1]
    print("ROC AUC score: ", roc_auc_score(y_test, y_proba))

    # False negative rate
    cm = confusion_matrix(y_test, y_pred)

    # tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    print("False negative rate: ", fn/(fn+tp))
    print("False positive rate: ", fp/(fp+tp))
    print("FPR + FNR: ", (fn/(fn+tp))+(fp/(fp+tp)))

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.plot(fpr, tpr, marker='.', label='LGBM')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    # Log-loss
    print("Log-loss: ", log_loss(y_test, y_proba))


def plot_feature_importance(importance,names,model_type):
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()
