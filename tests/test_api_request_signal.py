import pandas as pd
import numpy as np
import json
import requests
import traceback
from data_loader.data_loader import CsvDataLoader
from sklearn.metrics import accuracy_score

def test_loader_data_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/load-data/',
            data='{"data_path":"/Users/pwang/BenWork/OnlineML/onlineml/data/airline/airline_data.csv","label":"satisfaction"}',
            headers={'content-type': 'application/json'}
        )
        print(response.json())
        assert True
    except:
        traceback.print_stack()

def test_check_data_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/check-data/',
        )
        print(response.json())
        assert True
    except:
        traceback.print_stack()

def test_init_model_api():

    try:

        data = {
            "model_type": "sklearn random forest classifier",
            "model_params": {"n_estimators": "10","criterion": "gini","verbose": "1"}

        }

        model_params = {"n_estimators": "10","criterion": "gini","verbose": "1"}
        #'{"model_type":"sklearn random forest classifier", "model_params":"{"n_estimators": "10","criterion": "gini","verbose": "1"}"}'
        response = requests.post(
            url='http://127.0.0.1:5000/init-model/',
            data='{"model_type":"sklearn random forest classifier","model_params":{"n_estimators":10,"criterion":"gini","verbose":1}}',
            headers={'content-type': 'application/json'}
        )
        print(response.json())
        assert True
    except:
        traceback.print_exc()


def test_fit_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/fit/',
        )
        print(response.json())
        assert True
    except:
        pass


def test_predict_api():

    df = CsvDataLoader(data_path="/Users/pwang/BenWork/OnlineML/onlineml/data/airline/airline_data.csv").get_df(do_label_encoder=True)
    # df = pd.read_csv("/Users/pwang/BenWork/OnlineML/onlineml/data/airline/airline_data.csv")
    df = df.head(1000)
    y = df.pop('satisfaction')

    df_to_send = str(df.to_json())


    try:
        response = requests.post(
            url='http://127.0.0.1:5000/predict/',
            data=df_to_send,
            headers={'content-type': 'application/json'}
        )
        result = response.json()
        print(result)
        # result = json.loads(result)
        # print(np.asarray(result['array']))
        print(y.tolist())

        acc = accuracy_score(y.tolist(), result)
        print(acc)

        assert True

    except:
        traceback.print_exc()
