import pdb
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from datetime import date
from google.oauth2 import service_account
from google.cloud import bigquery
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBClassifier

from calculate_metrics import calculate_metrics, plot_feature_importance
from parameter_tuning import run_random_search

# Parameters
train_size = 0.8
random_seed = 123
n_iter_search = 20  # number of random hyper param search rounds
# Week number and year from last available week
week_number = 4
year = 2022
dataset = 'hollandbarrett'
column_dropped = ['OCH_ID', 'Y', 'date', 'week_number', 'year', 'quarter']
date = str(date.today()).replace('-', '_')

# TEST

# Cross validation hyperparameter distribution
param_dist = {'colsample_bytree': [0.64, 0.65, 0.66],
                'reg_alpha': uniform(1, 1.2),
                'reg_lambda': uniform(1, 1.4),
                'learning_rate': loguniform(0.005, 0.5),
                'max_depth': randint(1, 30),
                'num_leaves': randint(10, 200),
                'feature_fraction': uniform(0.1, 1.0),
                'subsample': uniform(0.1, 1.0)
                }

# Load credentials from Google Service Account in Google Cloud
credentials = service_account.Credentials.from_service_account_file('mimetic-union-338618-ea2a2d5e4148.json')
project_id = 'mimetic-union-338618'
# Get all rows before last Sunday
# query_train = f'SELECT * FROM hollandbarrett.final_table WHERE week_number != {week_number} \\
# ORDER BY RAND() LIMIT 1000000'
# Check how good performs if tested on trained data
query_train = f'SELECT * FROM hollandbarrett.final_table ORDER BY RAND() LIMIT 1000000'
# Last Sunday
query_test = f'SELECT * FROM hollandbarrett.final_table WHERE week_number = {week_number}'
bigquery_client = bigquery.Client(project=project_id, credentials=credentials)
destination = f'{dataset}.prediction_{date}'


def data_import():
    print("Start loading data...")

    # Run query to dataframe
    df_train = bigquery_client.query(query_train).to_dataframe()
    df_train.to_csv('df_train.csv')
    # df_train = pd.read_csv('df_train.csv').drop('Unnamed: 0', axis=1)
    df_test = bigquery_client.query(query_test).to_dataframe()
    df_test.to_csv('df_test.csv')
    # df_test = pd.read_csv('df_test.csv').drop('Unnamed: 0', axis=1)

    print(df_test)
    pdb.set_trace()

    return df_train, df_test


def train_model(df_train):
    df_1 = df_train[df_train['Y'] == 1]
    df_0 = df_train[df_train['Y'] == 0].sample(n=round((1*len(df_1))))

    df_train = df_0.append(df_1)

    x_train = df_train.drop(column_dropped, axis=1)
    y_train = df_train['Y']

    # Cross validation
    print("Start cross validation...")
    xgb = XGBClassifier()
    cv_results_random, opt_params = run_random_search(x_train, y_train, xgb, param_dist, n_iter_search)
    # opt_params = {'colsample_bytree': 0.64, 'feature_fraction': 0.9854297724729701,
    #               'learning_rate': 0.015518627558901415, 'max_depth': 6, 'num_leaves': 161,
    #               'reg_alpha': 1.5538158723169593, 'reg_lambda': 1.6055030807112185, 'subsample': 0.7827309401879335}
    xgb.set_params(**opt_params)

    # Fitting the model
    print("Start fitting the model...")
    xgb.fit(x_train, y_train)

    # Plot feature importances and correlations
    plot_feature_importance(xgb.feature_importances_, x_train.columns, 'XGBoost')
    # Plot correlations features
    sn.heatmap(x_train.corr())
    plt.show()

    # # Create pickle file from trained model
    # with open('model.pkl', 'wb') as model_file:
    #     pickle.dump(xgb, model_file)

    return xgb


def predict(clf, df_test):
    print("Start prediction...")
    x_test = df_test.drop(column_dropped, axis=1)
    y_test = df_test['Y']

    # Create prediction dataframe
    prediction = pd.DataFrame({
        'email': df_test['OCH_ID'],
        'proba': clf.predict_proba(x_test)[:, 1]
    })

    calculate_metrics(clf, y_test, x_test)

    return prediction


def prediction_export(prediction):
    print("Start exporting prediction...")

    # Write prediction to BigQuery table
    prediction.to_gbq(destination_table=destination, project_id=project_id, if_exists='replace'
                      , credentials=credentials)


def main():
    None
    # df_train, df_test = data_import()
    #
    # clf = train_model(df_train)
    #
    # prediction = predict(clf, df_test)

    # prediction_export(prediction)


if __name__ == '__main__':
    main()
