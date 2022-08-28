"""
Training functions for the apartment price estimate

Training provides functions to train a linear regression model on input data from ImmobilienScout24.de and
the following features of an apartment:

* 9 inputs for the regression model:
            wohnflaeche,
            zimmeranzahl,
            schlafzimmer,
            badezimmer,
            aufzug,
            balkon,
            denkmalobjekt,
            parkplatz,
            energieeffizienzklasse

* one output of the regression model:
            kaufpreis - apartment price as price offer from seller to potential owner of the apartment

use Linear Regression or Tweedy Regression
https://stats.stackexchange.com/questions/286709/do-you-need-to-split-data-for-linear-regression
https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
"""

import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

import pickle
import mlflow.sklearn
import argparse

import storage as storage

from dotenv import load_dotenv

# Loading environment variables for MLFLOW: MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_URI
# they are used implicitly by the ML Flow functions
load_dotenv()


def load_immo24_offers_from_csv_into_pandas_dataframe(path_to_file):
    """
    Load offers by ImmobilienScout24 for the German city "Leipzig" (CSV is prefiltered).

    All of them are offers for apartments in Leipzig to be sold.
    :param path_to_file: path to the CSV file
    :return: pandas dataframe with the selected columms
    """

    def custom_date_parser(x):
        """custom date parser to parse date columns like '2007m11' to a datetime value 2007-Nov-01"""
        return datetime.strptime(x, "%Ym%m")

    dataframe = pd.read_csv(path_to_file,
                            sep=";", decimal=",",
                            verbose=False,
                            on_bad_lines='warn',
                            low_memory=False,  # ensures proper data types, recommended by pandas during run
                            usecols=['obid', 'kaufpreis', 'wohnflaeche', 'zimmeranzahl',
                                     'schlafzimmer', 'badezimmer', 'aufzug', 'balkon', 'denkmalobjekt',
                                     'parkplatz', 'energieeffizienzklasse', 'duplicateid', 'adat', 'edat'],
                            parse_dates=['adat', 'edat'],
                            date_parser=custom_date_parser,
                            na_values={'zimmeranzahl': [-5, 0],
                                       'schlafzimmer': [-5, -9],
                                       'badezimmer': -9,
                                       'parkplatz': -9,
                                       'energieeffizienzklasse': -7,
                                       'duplicateid': -9}
                            )
    # print(dataframe.info())
    return dataframe


def load_feedback_from_csv_into_pandas_dataframe():
    """
    Load feedback from MinIO

    :return: pandas dataframe with feedback
    """

    # get feedback file from MinIO to local file
    client = storage.create_client()
    storage.get_feedback_from_minio(client)

    dataframe = pd.read_csv("feedback.csv",
                            sep=",", decimal=".",
                            verbose=False,
                            on_bad_lines='warn',
                            low_memory=False,  # ensures proper data types, recommended by pandas during run
                            )
    # print(dataframe.info())
    return dataframe


def preprocess_immo24_offers(dataframe, keep_since_year, remove_duplicates=True):
    """
    Remove offers, that are too old (before "keep_since_year").
    Also remove entries, that are considered duplicates of other entries, even though they may have a price update.
    :param dataframe:
    :param keep_since_year:
    :param remove_duplicates:
    :return:
    """
    # remove offers before the given year to keep
    dataframe = dataframe[(dataframe['adat'] >= f'{keep_since_year}-01-01')]

    if remove_duplicates:
        # duplicateid == NaN means "Line is *not* a duplicate of another one."
        # so the following line keeps the non-duplicates
        dataframe = dataframe[dataframe['duplicateid'].isnull()]

    # print(dataframe.info())
    return dataframe


def plot_input_data(df_immo24_offers, KEEP_SINCE_YEAR):
    """
    plot distributions of input data on the apartment offers

    :param df_immo24_offers:  list of apartments with features and price as a pandas data frame
    :param KEEP_SINCE_YEAR:   offers have been cut off before that year, used for labels in charts only
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=df_immo24_offers['wohnflaeche'], y=df_immo24_offers['kaufpreis'])
    plt.title(f"Kaufpreis zu Wohnfläche (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.xlabel("Wohnfläche (m²)")
    plt.ylabel("Kaufpreis (Mio EUR)")
    plt.xlim([0, 225])
    plt.ylim([0, 1.5e6])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=df_immo24_offers['zimmeranzahl'], y=df_immo24_offers['kaufpreis'])
    plt.title(f"Kaufpreis zu Zimmeranzahl (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.xlabel("Zimmeranzahl (Räume)")
    plt.ylabel("Kaufpreis (Mio EUR)")
    plt.xlim([0, 9])
    plt.ylim([0, 1.5e6])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=df_immo24_offers['schlafzimmer'], y=df_immo24_offers['kaufpreis'])
    plt.title(f"Kaufpreis zu Anz Schlafzimmer (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.xlabel("Schlafzimmeranzahl (Räume)")
    plt.ylabel("Kaufpreis (Mio EUR)")
    plt.xlim([-0.5, 5.5])
    plt.ylim([0, 1.5e6])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=df_immo24_offers['badezimmer'], y=df_immo24_offers['kaufpreis'])
    plt.title(f"Kaufpreis zu Anz Badezimmer (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.xlabel("Bäderanzahl (Räume)")
    plt.ylabel("Kaufpreis (Mio EUR)")
    plt.xlim([0, 3.5])
    plt.ylim([0, 1.5e6])
    plt.show()

    n_bins = 100
    plt.figure(figsize=(10, 6))
    plt.hist(df_immo24_offers.query('aufzug == 0')['kaufpreis'], bins=n_bins, alpha=0.5, label="Kein Aufzug")
    plt.hist(df_immo24_offers.query('aufzug == 1')['kaufpreis'], bins=n_bins, alpha=0.5, label="Aufzug vorhanden")
    plt.xlabel("Kaufpreis (Mio EUR)")
    plt.ylabel("Anzahl Wohnungen")
    plt.title(f"Verteilung Preis für Aufzug (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(df_immo24_offers.query('balkon == 0')['kaufpreis'], bins=n_bins, alpha=0.5, label="Kein Balkon")
    plt.hist(df_immo24_offers.query('balkon == 1')['kaufpreis'], bins=n_bins, alpha=0.5, label="Balkon vorhanden")
    plt.xlabel("Kaufpreis (Mio EUR)")
    plt.ylabel("Anzahl Wohnungen")
    plt.title(f"Verteilung Preis für Balkon (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(df_immo24_offers.query('denkmalobjekt == 0')['kaufpreis'], bins=n_bins, alpha=0.5,
             label="Kein Denkmalobjekt")
    plt.hist(df_immo24_offers.query('denkmalobjekt == 1')['kaufpreis'], bins=n_bins, alpha=0.5, label="Denkmalobjekt")
    plt.xlabel("Kaufpreis (Mio EUR)")
    plt.ylabel("Anzahl Wohnungen")
    plt.title(f"Verteilung Preis für Denkmalobjekt (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(df_immo24_offers.query('parkplatz == 0')['kaufpreis'], bins=n_bins, alpha=0.5, label="Kein Parkplatz")
    plt.hist(df_immo24_offers.query('parkplatz == 1')['kaufpreis'], bins=n_bins, alpha=0.5, label="Parkplatz vorhanden")
    plt.xlabel("Kaufpreis (Mio EUR)")
    plt.ylabel("Anzahl Wohnungen")
    plt.title(f"Verteilung Preis für Parkplatz (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.legend(loc='upper right')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=df_immo24_offers['energieeffizienzklasse'], y=df_immo24_offers['kaufpreis'])
    plt.title(f"Kaufpreis zu Energieeffizienzklasse (ImmobilienScout24 für Wohnungen in Leipzig ab {KEEP_SINCE_YEAR})")
    plt.xlabel("Energieeffizienzklasse (1=A, 8=H)")
    plt.ylabel("Kaufpreis (Mio EUR)")
    plt.xlim([0, 8.5])
    plt.ylim([0, 1.5e6])
    plt.show()

    # plot histogram of each feature
    # df_immo24_offers.hist('kaufpreis', bins=100)
    # plt.show()
    # df_immo24_offers.hist('wohnflaeche', bins=100)
    # plt.show()
    # df_immo24_offers.hist('zimmeranzahl', bins=100)
    # plt.show()
    # df_immo24_offers.hist('schlafzimmer', bins=100)
    # plt.show()
    # df_immo24_offers.hist('badezimmer', bins=100)
    # plt.show()
    # df_immo24_offers.hist('aufzug', bins=100)
    # plt.show()
    # df_immo24_offers.hist('balkon', bins=100)
    # plt.show()
    # df_immo24_offers.hist('denkmalobjekt', bins=100)
    # plt.show()
    # df_immo24_offers.hist('parkplatz', bins=100)
    # plt.show()
    # df_immo24_offers.hist('energieeffizienzklasse', bins=100)
    # plt.show()


def train_regression_model(df_offers, df_feedback=None):
    """
    train a regression model on a number of training data
    training data is a list of apartments with features and price, and therefore labeled data

    :param df_offers: list of apartments with features and price as a pandas data frame
    :param df_feedback: optional dataframe with feedback data
    :return: regression_model as sklearn object
    """
    # drop unnecessary columns
    df_offers = df_offers.drop(['obid', 'duplicateid', 'adat', 'edat'], axis=1)

    if df_feedback is not None:
        df_offers = pandas.concat([df_offers, df_feedback], ignore_index=True)

    # prepare output of the regression model
    y = df_offers['kaufpreis']

    # prepare input of the regression model
    X = df_offers.drop(['kaufpreis'], axis=1)

    # impute NaN with values that make sense and do not skew the data set
    imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer_mean.fit_transform(X)

    regression_model = LinearRegression().fit(X_imputed, y)

    print()
    print("Results of the LinearRegression model:")
    print(f"""Score: {regression_model.score(X_imputed, y)}""")
    print(f"""Intercept (Offset): {regression_model.intercept_}""")
    feature_names = ['wohnflaeche', 'zimmeranzahl',
                     'schlafzimmer', 'badezimmer', 'aufzug', 'balkon', 'denkmalobjekt',
                     'parkplatz', 'energieeffizienzklasse']
    print(f"""Coefficients: {list(zip(feature_names, regression_model.coef_))}""")
    return regression_model


def save_regression_model_to_file(regression_model, file_name):
    """save the model to disk"""
    pickle.dump(regression_model, open(file_name, 'wb'))


def save_regression_model_to_model_store(regression_model):
    """save the model to model store in ML Flow"""
    # log model
    result = mlflow.sklearn.log_model(sk_model=regression_model,
                                      artifact_path='group7-linear-regression-model',
                                      registered_model_name='group7-linear-regression-model')
    print(result)


def main() -> int:
    """
    Training pipeline: load raw data from ImmobilienScout24 (CSV), preprocess, train regression model, save to file
    """
    parser = argparse.ArgumentParser(description='Training pipeline with optional feedback processing')
    parser.add_argument("-f", action='store_true', help="include feedback data in training")
    args = parser.parse_args()

    # print(os.getcwd()) => "c:/.../se4ai-2022-7/apartment_price_estimate"
    IMMO24_DATA_FILE = '../data/CampusFile_Wohnungskauf_Leipzig.csv'
    KEEP_SINCE_YEAR = 2020

    df_immo24_offers = load_immo24_offers_from_csv_into_pandas_dataframe(IMMO24_DATA_FILE)
    df_immo24_offers = preprocess_immo24_offers(df_immo24_offers,
                                                keep_since_year=KEEP_SINCE_YEAR, remove_duplicates=True)
    pd.options.display.max_columns = df_immo24_offers.shape[1]
    # print(df_immo24_offers.describe())
    plot_input_data(df_immo24_offers, KEEP_SINCE_YEAR)

    if args.f:
        print("Training on Immoscout24 data *** plus feedback data ***.")
        df_feedback = load_feedback_from_csv_into_pandas_dataframe()
        regression_model = train_regression_model(df_immo24_offers, df_feedback)
    else:
        print("Training on Immoscout24 data only. Feedback is neglected.")
        regression_model = train_regression_model(df_immo24_offers)

    save_regression_model_to_file(regression_model, 'model/lpz_apt_prices_regression_model.pickle')
    save_regression_model_to_model_store(regression_model)
    return 0


if __name__ == '__main__':
    sys.exit(main())
