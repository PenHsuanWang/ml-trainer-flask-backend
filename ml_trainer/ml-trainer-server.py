from flask import Flask, request, Response, abort
import time
import traceback

import pandas

from data_loader.data_loader import CsvDataLoader

# app = Flask(__name__)


class ModelTrainerServer:

    def __init__(self):
        self._data_loader = None
        self._model = None

        self.app = Flask(__name__)


    def ml_trainer_hello_message(self):  # put application's code here
        # return render_template('home/index.html')
        return 'Hello, this is the app for ml model trainer implement by callback function.<br>' \
               'Provide the data path and initialization the model configuration to do model training.'

    def load_data(self):
        print("going to test dataloader")

        try:
            data_path = request.get_json()['data_path']
            print("target data path is {}".format(data_path))

            if data_path.split(".")[-1] == 'csv':
                print("data format is csv")
                self._data_loader = CsvDataLoader(data_path=data_path)
                df = self._data_loader.get_df(do_label_encoder=True)
                print(df.head(5))
                print("load data success")

        except:
            traceback.print_exc()

        return Response(
            status=200
        )


    def run(self):
        self.app.add_url_rule(rule='/', endpoint='ml_training_message', view_func=self.ml_trainer_hello_message)
        self.app.add_url_rule(rule='/load-data/', endpoint='load_data', view_func=self.load_data, methods=['POST'])
        print("Start flask server")
        self.app.run()


    #
    # @app.route('/init-model')
    # def init_model():
    #     print("constructing model")
    #     time.sleep(1)
    #     print("finished of model initialization.")
    #
    #     return Response(
    #         status=200
    #     )
    #
    # @app.route('/fit')
    # def fit():
    #     print("model fitting")
    #     time.sleep(10)
    #     print("finished of model training")
    #
    #     return Response(
    #         status=200
    #     )



if __name__ == '__main__':

    server = ModelTrainerServer()
    server.run()


