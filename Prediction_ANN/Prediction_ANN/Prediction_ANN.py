import pandas as pd
import numpy as np
import descartes
import geopandas as gpd
import contextlib as ctx
from matplotlib import pyplot
from shapely.geometry import Point, Polygon
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from shapely.geometry import Point, Polygon
from pandas import ExcelWriter
from pandas import ExcelFile
from pathlib import Path  
# Extract predicted and true values
'''
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
for index in range(1):
        df = dataset_GP.iloc[1158, 782:]
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
        last_predicted
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
'''

df = pd.read_excel('predicted.xlsx', index_col=0)
df
# Plot points on map
geometry = [Point(xy) for xy in zip(df["lng"], df["lat"])]

# concat multible shapefiles into one gpd df
folder = Path("shp")

gdf = pd.concat([
    gpd.read_file(shp)
    for shp in folder.glob("*.shp")
], sort=False).pipe(gpd.GeoDataFrame)
gdf.to_crs({'proj': 'merc'})

geo_df = gpd.GeoDataFrame(df,
                          crs= 'merc',
                          geometry = geometry)
geo_df.head()

geo_df['true values'][0] % geo_df['predicted values'][0]
-0.0281 <= -0.0004
(-0.0211 % -0.020993)
(-0.0064 % 0.006095)

(geo_df['predicted values'] > 0) 
(geo_df['predicted values'] > -0.02) & (geo_df['predicted values'] < 0)
(geo_df['predicted values'] > -0.09) & (geo_df['predicted values'] <= -0.02)
(geo_df['predicted values'] <= -0.021) 
geo_df['predicted values']
-4 > -6

xlim = ([12.030, 12.0475])
ylim = ([57.615, 57.640])

fig, (ax1, ax2) = pyplot.subplots(1,2, sharey=True, figsize=(15,15))

ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax1.set_title('True values', fontsize='x-large')
ax2.set_title('Predicted values', fontsize='x-large')
gdf.plot(ax = ax1, alpha=0.8, zorder=0)
gdf.plot(ax = ax2, alpha=0.8, zorder=0)

dot_size = 1

# Points for ax1
#geo_df[geo_df['true values'] <= -0.004].plot(ax = ax1, markersize = 2, color = 'red', marker = "o", label = "Negativ förändring", zorder=5)
geo_df[(geo_df['true values'] > 0)].plot(ax = ax1, markersize = dot_size, color = 'green', marker = "o", label = "Högst höjning", zorder=6)
geo_df[(geo_df['true values'] > -0.02) & (geo_df['true values'] < 0)].plot(ax = ax1, markersize = dot_size, color = 'yellow', marker = "o", label = "Mindre sättning", zorder=4)
geo_df[(geo_df['true values'] > -0.09) & (geo_df['true values'] <= -0.02)].plot(ax = ax1, markersize = dot_size, color = 'orange', marker = "o", label = "Medel sättning", zorder=5)
geo_df[(geo_df['true values'] <= -0.021)].plot(ax = ax1, markersize = dot_size, color = 'red', marker = "^", label = "Högst sättning", zorder=6)

# Points for ax2
geo_df[(geo_df['predicted values'] > 0)].plot(ax = ax2, markersize = dot_size, color = 'green', marker = "o", label = "Högst höjning", zorder=6)
geo_df[(geo_df['predicted values'] > -0.02) & (geo_df['predicted values'] < 0)].plot(ax = ax2, markersize = dot_size, color = 'yellow', marker = "o", label = "Mindre sättning", zorder=4)
geo_df[(geo_df['predicted values'] > -0.09) & (geo_df['predicted values'] <= -0.02)].plot(ax = ax2, markersize = dot_size, color = 'orange', marker = "o", label = "Medel sättning", zorder=5)
geo_df[(geo_df['predicted values'] <= -0.021)].plot(ax = ax2, markersize = dot_size, color = 'red', marker = "^", label = "Högst sättning", zorder=6)
#geo_df[geo_df['true values'] % geo_df['predicted values'] < -0.004].plot(ax = ax, markersize = 10, color = 'red', marker = "o", label = "Negativ förändring", zorder=5)
#geo_df[geo_df['true values'] % geo_df['predicted values'] >= -0.004].plot(ax = ax, markersize = 10, color = 'yellow', marker = "^", label = "Ingen förändring", zorder=5)

pyplot.legend(prop={'size': 15})
pyplot.show()


# street = gpd.read_file("shape/Exjobb.shp")
# street = gpd.GeoDataFrame(street)
# fig, ax = pyplot.subplots(figsize = (15,15))
# street.plot(ax = ax)
# pyplot.show()
        


