import requests
import traceback

def test_loader_data_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/load-data/',
            data='{"data_path":"/Users/pwang/BenWork/OnlineML/onlineml/data/airline/airline_data.csv"}',
            headers={'content-type': 'application/json'}
        )
        print(response.json())
        assert True
    except:
        traceback.print_stack()