import os
from flask import Flask, request, Response
import pandas as pd
import pickle

from rocket  import Rocket

import inflection

#talvez tenha q ser um caminho mais curto
model = pickle.load(open('model/model_house.pkl', 'rb'))

app = Flask( __name__ )

@app.route('/rocket/predict', methods=['POST'])

def rocket_predict():

    test_json = request.get_json()

    if test_json: #there is data

        if isinstance( test_json, dict): #Unique example

            test_raw = pd.DataFrame( test_json, index=[0])

        else: # multiple example

            test_raw = pd.DataFrame( test_json, columns = test_json[0].keys() )

        #Instantiate Rocket Class
        pipeline = Rocket()

        #data cleaning
        df1 =  pipeline.data_cleaning( test_raw )

        #data preparation
        df2 = pipeline.data_preparation(df1)

        #prediction
        df_response = pipeline.get_prediction(model, test_raw, df2)

        return df_response

    else:
        return Response( '{}', status=200, mimetype='application/json')

if __name__ == '__main__':

    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port=port)
