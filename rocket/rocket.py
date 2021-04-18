import pickle
import inflection
import numpy as np
import pandas as pd

class Rocket( object):
    def __init__(self):

        self.home_path = ''

        self.sqft_living =  pickle.load( open(self.home_path  + 'parametrer/sqft_living.pkl', 'rb'))

        self.sqft_lot =     pickle.load(open(self.home_path  + 'parametrer/sqft_lot.pkl', 'rb'))

        self.sqft_above =  pickle.load(open(self.home_path  + 'parametrer/sqft_above.pkl', 'rb'))

        self.yr_built =  pickle.load(open(self.home_path  + 'parametrer/yr_built.pkl', 'rb'))

        self.sqft_lot15 =  pickle.load(open(self.home_path  + 'parametrer/sqft_lot15.pkl', 'rb'))

        self.lat =  pickle.load(open(self.home_path  + 'parametrer/lat.pkl', 'rb'))

        self.long = pickle.load(open(self.home_path  + 'parametrer/long.pkl', 'rb'))

        self.sqft_living15 = pickle.load(open(self.home_path  + 'parametrer/sqft_living15.pkl', 'rb'))

    def data_cleaning(self, df1 ):

        df1.dropna(axis=0, inplace=True)

        df1 = df1[~(df1['bedrooms'] == 0)]
        df1 = df1[~(df1['bathrooms'] == 0)]

        return df1

    def data_preparation(self, df2):
        #Rescaling

        df2['sqft_living'] = self.sqft_living.fit_transform( df2[['sqft_living']].values)

        df2['sqft_lot'] = self.sqft_living.fit_transform(df2[['sqft_lot']].values)

        df2['sqft_above'] = self.sqft_living.fit_transform(df2[['sqft_above']].values)

        df2['yr_built'] = self.sqft_living.fit_transform(df2[['yr_built']].values)

        df2['sqft_lot15'] = self.sqft_living.fit_transform(df2[['sqft_lot15']].values)

        df2['lat'] = self.sqft_living.fit_transform(df2[['lat']].values)

        df2['long'] = self.sqft_living.fit_transform(df2[['long']].values)

        df2['sqft_living15'] = self.sqft_living.fit_transform(df2[['sqft_living15']].values)

        cols_selected = [
            'sqft_living',
            'sqft_lot',
            'view',
            'grade',
            'sqft_above',
            'yr_built',
            'lat',
            'long',
            'sqft_living15',
            'sqft_lot15']

        return df2[cols_selected]

    def get_prediction ( self, model, original_data, test_data  ):

        # prediction
        pred = model.predict(test_data)

        #join pred into the original data
        original_data['prediction'] = np.exp( pd.DataFrame(pred) )

        return original_data.to_json( orient='records')