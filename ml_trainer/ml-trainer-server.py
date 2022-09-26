from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, Response, abort
import time
import json
from json import JSONEncoder
import traceback

import pandas
import numpy

from data_loader.data_loader import CsvDataLoader

# app = Flask(__name__)


class ModelTrainerServer:

    def __init__(self):
        self._data_loader = None
        self._model = None
        self._df_x = None
        self._y = None

        self.app = Flask(__name__)


    def ml_trainer_hello_message(self):  # put application's code here
        # return render_template('home/index.html')
        return 'Hello, this is the app for ml model trainer implement by callback function.<br>' \
               'Provide the data path and initialization the model configuration to do model training.'

    def load_data(self):
        print("going to test dataloader")

        try:
            data_path = request.get_json()['data_path']
            label_name = request.get_json()['label']
            print("target data path is {}".format(data_path))

            if data_path.split(".")[-1] == 'csv':
                print("data format is csv")
                self._data_loader = CsvDataLoader(data_path=data_path)

                self._df_x = self._data_loader.get_df(do_label_encoder=True)
                self._y = self._df_x.pop(label_name)

                print(self._df_x.head(5))
                print("load data success")

                return Response(
                    json.dumps(
                        'load {} data success'.format(data_path)
                    ),
                    status=200,
                    headers={'content-type': 'application/json'}
                )

        except:
            traceback.print_exc()

        return Response(
            status=200
        )

    def check_data(self):
        print(self._df_x.head(5))
        print(self._y.head(5))

        return Response(
            json.dumps(
                str(self._df_x.head(5)) +
                '\n' +
                str(self._y.head(5))
            ),
            status=200,
            headers={'content-type':'application/json'}
        )

    def init_model(self):

        print("going to initialized the model")

        try:
            model_type = request.get_json()['model_type']
            model_params = request.get_json()['model_params']

            print(model_type)
            print(model_params)

            model = None

            if model_type == 'sklearn random forest classifier':

                try:
                    model = RandomForestClassifier(
                        **model_params
                    )
                    print(model)
                except:
                    traceback.print_exc()

            if model is not None:
                self._model = model
                print("finish of model initialization")

                return Response(
                    json.dumps("finish of model initialization"),
                    status=200,
                    headers={'content-type': 'application/json'}
                )

        except:
            traceback.print_exc()


    def fit(self):

        print("going to fit the model")

        try:
            self._model.fit(self._df_x, self._y)
            print("trained model successfully")

            return Response(
                status=200
            )

        except:
            traceback.print_exc()
            print('Unexpect error happen when doing model fitting')


    def predict(self):

        class NumpyArrayEncoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, numpy.ndarray):
                    return obj.tolist()
                return JSONEncoder.default(self, obj)

        print("going to do data prediction")

        try:
            data = request.get_json()

            data = json.dumps(data)
            df = pandas.read_json(data)
            print(df)
            print(df.head())

            result = self._model.predict(df)
            print(result)
            return Response(
                json.dumps(result, cls=NumpyArrayEncoder),
                status=200,
                headers={'content-type': 'application/json'}
            )

        except:
            traceback.print_exc()



    def run(self):
        self.app.add_url_rule(rule='/', endpoint='ml_training_message', view_func=self.ml_trainer_hello_message)
        self.app.add_url_rule(rule='/load-data/', endpoint='load_data', view_func=self.load_data, methods=['POST'])
        self.app.add_url_rule(rule='/check-data/', endpoint='check_data', view_func=self.check_data, methods=['POST'])
        self.app.add_url_rule(rule='/init-model/', endpoint='init_model', view_func=self.init_model, methods=['POST'])
        self.app.add_url_rule(rule='/fit/', endpoint='fit', view_func=self.fit, methods=['POST'])
        self.app.add_url_rule(rule='/predict/', endpoint='predict', view_func=self.predict, methods=['POST'])
        print("Start flask server")
        self.app.run()





if __name__ == '__main__':

    server = ModelTrainerServer()
    server.run()


