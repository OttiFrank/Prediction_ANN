import pandas as pd
import numpy as np
from matplotlib import pyplot
from shapely.geometry import Point, Polygon
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import contextlib as ctx
from pandas import ExcelWriter
from pandas import ExcelFile

url='http://users.du.se/~h16wilwi/gik258/data/ANN-interpolerad.xlsx'
dataset = pd.read_excel(url, skiprows=3)

#lat_long_only = dataset[['pnt_lat','pnt_lon']]
lat_long_only = dataset.iloc[:1159, 2:4]
lat_long_only
dataset = dataset.drop(['pnt_id', 'pnt_lat', 'pnt_lon', 'pnt_demheight', 'pnt_height', 'pnt_quality', 'pnt_linear'], axis=1)

dataset.set_index('index', inplace=True)
dataset = dataset.drop(['Daggp_mean', 'TYtaDaggp_mean'])

# Ground data
dataset_GP = dataset.iloc[:1159, :]
len(dataset_GP)
# Weather data
dataset_W = dataset.iloc[1159:, :]
len(dataset_W)
# Transpose dataset
dataset_W = dataset_W.transpose()
#dataset_GP = dataset_GP.transpose()
dataset_GP
len(dataset_GP)
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

n_days = 1
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

pred_list = list()
# For each data point
len(dataset_GP)
for index in range(len(dataset_GP)-1):
        df = dataset_GP.iloc[index, 782:]
        df = pd.concat([df, dataset_W], axis=1)
        print('index: {}'.format(index))

        values = df.astype('float64')
        #normalize features
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = series_to_supervised(scaled, n_days,1 )
        values = reframed.values
        test_X, test_y = values[:, :n_obs], values[:, -n_features]
        test_X = test_X.reshape(test_X.shape[0], n_days, n_features)
        # Compile ANN
        loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        # Make prediction
        prediction = loaded_model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
        # invert scaling for forecast
        inv_yhat = np.concatenate((prediction, test_X[:, -(n_features-1):]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        last_predicted = inv_yhat[-1]

        pred_list.append(last_predicted)
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        
        inv_y = inv_y[:,0]
        last_true_value = inv_y[-1]
        if (index % 5 == 0):
            print(pred_list)



pred_list = pd.DataFrame([pred_list])
pred_list = pred_list.transpose()
pred_list.reset_index(level=0, inplace=True)
pred_list.set_index('index')
true_values = pd.DataFrame([dataset.iloc[:1159,-1]])
true_values = true_values.transpose()
true_values

df = pd.concat([true_values, pred_list], axis=1, sort=False)
lat_long_only
result = pd.concat([lat_long_only, df], axis=1, sort=False)
result
result = result.drop(['index'], axis=1)
result.columns = ["lat", "lon", 'true values', 'predicted values']
#result = result.rename(index=str, columns={result.columns: 'True values', 0 : 'Predicted'})
print(result)


writer = ExcelWriter('predicted.xlsx', engine='xlsxwriter')
writer.book.use_zip64()
result.to_excel(writer, sheet_name="Blad1")
writer.save()

# my_list = map(lambda x: x[0], prediction)
# series = pd.Series(my_list)
# series.hist(align='mid')
# pyplot.show()


# street = gpd.read_file("shape/Exjobb.shp")
# street = gpd.GeoDataFrame(street)
# fig, ax = pyplot.subplots(figsize = (15,15))
# street.plot(ax = ax)
# pyplot.show()
        


