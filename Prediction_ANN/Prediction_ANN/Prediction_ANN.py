import pandas as pd
from matplotlib import pyplot
from shapely.geometry import Point, Polygon
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon

url='http://users.du.se/~h16wilwi/gik258/data/ANN-interpolerad.xlsx'
dataset = pd.read_excel(url, skiprows=3)

dataset = dataset.drop(['pnt_id', 'pnt_lat', 'pnt_lon', 'pnt_demheight', 'pnt_height', 'pnt_quality', 'pnt_linear'], axis=1)

dataset.set_index('index', inplace=True)
dataset = dataset.drop(['Daggp_mean', 'TYtaDaggp_mean'])

dataset_GP = dataset.iloc[:1159, :]
dataset_W = dataset.iloc[1159:, :]

# Transpose dataset
dataset_W = dataset_W.transpose()
dataset_GP = dataset_GP.transpose()

# Convert series to supervised learning
def series_to_supervised(values, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(values) is list else values.shape[1]
    df = pd.DataFrame(values)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forcast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)]
    # PPut it all together
    agg = pd.concat(cols, axis = 1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan: 
        agg.dropna(inplace=True)
    return agg

n_days = 3
n_features = 11
n_obs = n_days * n_features
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

prediction_list = list()

n_days = 3
n_features = 11
n_obs = n_days * n_features

# For each data point
for index in range(1):
        df = dataset_GP.iloc[:, index]
        #df = pd.concat([df,dataset_W], sort=False)
        df = pd.concat([df, dataset_W], axis=1)
        print('index: {}'.format(index))

        values = df.astype('float32')
        #normalize features
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = series_to_supervised(scaled, n_days,1 )
        values = reframed.values
        val = values[:, :n_obs]
        val = val.reshape(val.shape[0], n_days, n_features)
        # Compile ANN
        loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        # Make prediction
        prediction = loaded_model.predict_proba(val)
        prediction_list.append(prediction)
        my_list = map(lambda x: x[0], prediction)
        series = pd.Series(my_list)
        series.hist(align='mid')
        pyplot.show()



street = gpd.read_file("shape/exjobb.shp")
street = gpd.GeoDataFrame(street)
fig, ax = pyplot.subplots(figsize = (15,15))
street.plot(ax = ax)
pyplot.show()
        


