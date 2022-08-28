"""
Inference functions for the apartment price estimate

This file offers the function for actually doing the apartment price estimate from a pre-trained model,
that uses linear regression.
* The regression model is typically loaded from a MLFlow model registry but is also possible to load from a file.
* Inference then uses a simple predict() on the regression model with a feature vector for an apartment.
"""

import pickle
import numpy as np
import mlflow.sklearn
from dotenv import load_dotenv

# Loading environment variables for MLFLOW: MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_URI
# they are used implicitly by the ML Flow functions
load_dotenv()


def load_regression_model_from_file(file_name) -> object:
    """load a regression model from a file
    the file must have been saved before with train.py
    """
    loaded_model = pickle.load(open(file_name, 'rb'))
    return loaded_model


def load_regression_model_from_model_store() -> object:
    """load the model from model store in ML Flow"""
    loaded_model = mlflow.sklearn.load_model("models:/group7-linear-regression-model/Production")
    return loaded_model


def predict_price_on_regression_model(regression_model, apartment_features) -> int:
    """
    Predict the price of an apartment in Leipzig

    :param regression_model: a regression_model that must be loaded beforehand via load_regression_model
    :param apartment_features: an input vector as numpy array of exactly 9 float values
        wohnflaeche   : apartment living area in square meters
        zimmeranzahl  : 1 to 8 or 1.5 or 2.5 for half rooms
        schlafzimmer  : number of bed rooms: 1, 2, 3, 4, 5
        badezimmer    : number of bath rooms: 1, 2, 3
        aufzug        : 0 = no elevator, 1 = elevator present
        balkon        : 0 = no balcony, 1 = balcony present
        denkmalobjekt : 1 = protected as a historic monument (listed on historic register), 0 not protected
        parkplatz     : 0 = no parking space included , 1 = parking space available or included
        energieeffizienzklasse : energy efficiency: 1, 2, 3, 4, 5, 6, 7, 8, = A (best), B, C, D, E, F, G, H (worst)

    :return: predicted price in EUR
    """
    result = regression_model.predict(apartment_features)
    # item() to get rid of surrounding np.array
    # rounding to cut decimals, which do not make sense anyways
    return round(result.item())


def main():
    """
    Some simple test code to show usage of the defined functions.
    """
    # regression_model = load_regression_model_from_file("model/lpz_apt_prices_regression_model.pickle")
    regression_model = load_regression_model_from_model_store()

    sample_apartments = [np.array([[50,  2, 1, 1, 0, 0, 0, 1, 4]]),
                         np.array([[50, 2.5, 1, 1, 0, 0, 0, 1, 4]]),
                         np.array([[75,  3, 1, 1, 0, 0, 0, 1, 4]]),
                         np.array([[100, 4, 2, 1, 0, 0, 0, 1, 4]]),
                         np.array([[100, 4, 2, 2, 0, 0, 0, 1, 4]]),
                         np.array([[100, 4, 2, 3, 0, 0, 0, 1, 4]]),
                         np.array([[200, 6, 3, 2, 1, 1, 1, 1, 7]]),
                         np.array([[200, 6, 3, 2, 1, 1, 1, 1, 2]]),
                         ]
    for x in sample_apartments:
        apartment_price_estimate = predict_price_on_regression_model(regression_model, x)
        price = f'{apartment_price_estimate:,}'
        print(f"""The prediction on {x} is {price} EUR.""")


if __name__ == '__main__':
    main()
