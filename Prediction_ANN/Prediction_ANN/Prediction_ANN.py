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

# Creates a file with the last predicted and true value for all InSAR-measurments
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
dataset_GP
# Weather data
dataset_W = dataset.iloc[1159:, :]
# Transpose dataset
dataset_W = dataset_W.transpose()
#dataset_GP = dataset_GP.transpose()
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

prediction_list = list()
# from column 927 (points)
i = 927

len(dataset_GP)


# For each data point
for index  in range(1159):        
        df = dataset_GP.iloc[index, 782:]
        df = pd.concat([df, dataset_W], axis=1)
        df
        print('index: {}'.format(index))
        values = df.astype('float64')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = series_to_supervised(scaled, n_days,1 )
        values = reframed.values
        val, val_test = values[:, :n_obs], values[:, -n_features]
        val = val.reshape(val.shape[0], n_days, n_features)
        # Compile ANN
        #loaded_model.compile(loss='mse', optimizer='adam')

        # Evaluate the model
        scores = loaded_model.predict(val)

        # Make prediction
        prediction = loaded_model.predict(val)
        test_X = val.reshape((val.shape[0], n_days*n_features))
        # invert scaling for forecast
        inv_yhat = np.concatenate((prediction, test_X[:, -(n_features-1):]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        last_predicted = inv_yhat[-1]
        prediction_list.append(last_predicted)
        i += 1


pred_list = pd.DataFrame([prediction_list])
pred_list = pred_list.transpose()
pred_list
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

pred_list = pred_list

inv_yhat
prediction
prediction_list[0]
pred_list = []
summary = list()
len(prediction_list[0])
'''



for j in range(len(prediction_list)):
    for i in range (len(prediction_list[j])):
        pred_list.append(prediction_list[0][i][0])
        summary.append((prediction_list[0][i][0] * 100))

my_list = map(lambda x: x[0], pred_list)
series = pd.Series(pred_list)
series.hist()
pyplot.show()

'''
# Plot our points on a map
url = "http://users.du.se/~h15marle/GIK258_Examensarbete/Data/railway_data.csv"
all_points = pd.read_csv(url)
our_points = pd.read_excel('predicted.xlsx', index_col=0)
our_points = our_points.iloc[:, 0:2]
all_points = all_points.iloc[:, 1:3]
all_points

all_geometry = [Point(xy) for xy in zip(all_points["pnt_lat"], all_points["pnt_lon"])]
our_geometry = [Point(xy) for xy in zip(our_points["lat"], our_points["lng"])]

# concat multible shapefiles into one gpd df
folder = Path("shp")
print("---- Letar efter och concat av shape-filer ---- ")
gdf = pd.concat([
    gpd.read_file(shp)
    for shp in folder.glob("*.shp")
], sort=False).pipe(gpd.GeoDataFrame)

print("--- Startar skapandet av GeoDataFrame -----")
geo_df_all = gpd.GeoDataFrame(all_points,
                          crs= 'merc',
                          geometry = all_geometry)
geo_df_our = gpd.GeoDataFrame(our_points,
                          crs= 'merc',
                          geometry = our_geometry)

def plot_map():
    # Limit map size
    xlim = ([12.030, 12.0475])
    ylim = ([57.615, 57.640])
    
    # dot size
    dot_size = 1

    fig, (ax1, ax2) = pyplot.subplots(1,2, sharey=True, figsize=(15,15))

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax1.set_title('All points', fontsize='xx-large')
    ax2.set_title('Our points', fontsize='xx-large')
    print("--- Plottar ut första kartan ----")
    gdf.plot(ax = ax1, alpha=0.8, zorder=0)
    print("--- Plottar ut andra kartan ----")
    gdf.plot(ax = ax2, alpha=0.8, zorder=0)    
    geo_df_all
    geo_df_all['geometry'].plot(ax = ax1, markersize = dot_size, color = 'gold', marker = "o", zorder=6)
    geo_df_our['geometry'].plot(ax = ax2, markersize = dot_size, color = 'gold', marker = "o", zorder=6)
    pyplot.show()

plot_map()
'''

# Calculates accuracy for each last measurement from "prediction.xlsx"
'''
df = pd.read_excel('predicted.xlsx', index_col=0)
df
df_true, df_pred = df, df
df_true = df_true.drop('predicted values', axis=1)
df_pred = df_pred.drop('true values', axis=1)

percentage_values = list()
for i in range(len(df_pred)): 
    pred, true = df_pred.iloc[i,2], df_true.iloc[i,2]
    pred = abs(pred)
    true = abs(true)
    if pred > true:        
        percentage = true / pred
    else:
        percentage = pred / true
    percentage_values.append(percentage)

percentage_values

my_list = pd.Series(percentage_values)
my_list = my_list.replace(-np.inf, 0)
fig = pyplot.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_title('Accuracy per point in %')
pyplot.xticks(np.arange(0, 1, step=0.05))
my_list.hist()
pyplot.show()
'''
# END

# Plot points on map
geometry = [Point(xy) for xy in zip(df["lng"], df["lat"])]

# concat multible shapefiles into one gpd df
folder = Path("shp")
print("---- Letar efter och concat av shape-filer ---- ")
gdf = pd.concat([
    gpd.read_file(shp)
    for shp in folder.glob("*.shp")
], sort=False).pipe(gpd.GeoDataFrame)

print("--- Startar skapandet av GeoDataFrame -----")
geo_df_true = gpd.GeoDataFrame(df_true,
                          crs= 'merc',
                          geometry = geometry)
geo_df_pred = gpd.GeoDataFrame(df_pred,
                          crs= 'merc',
                          geometry = geometry)
print(" ----------------------------------------- ")
def plot_map():
    # Limit map size
    xlim = ([12.030, 12.0475])
    ylim = ([57.615, 57.640])
    
    # dot size
    dot_size = 2

    fig, (ax1, ax2) = pyplot.subplots(1,2, sharey=True, figsize=(15,15))

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax1.set_title('True values', fontsize='xx-large')
    ax2.set_title('Predicted values', fontsize='xx-large')
    print("--- Plottar ut första kartan ----")
    gdf.plot(ax = ax1, alpha=0.8, zorder=0)
    print("--- Plottar ut andra kartan ----")
    gdf.plot(ax = ax2, alpha=0.8, zorder=0)    

    print("---- Påbörjar uträkningarna av plottar -----")
    # Points for ax1
    geo_df_true[(geo_df_true['true values'] > 0.001)].plot(ax = ax1, markersize = dot_size, color = 'blue', marker = "*", label="> 1mm", zorder=6)
    geo_df_true[(geo_df_true['true values'] >= -0.001) & (geo_df_true['true values'] <= 0.001)].plot(ax = ax1, markersize = dot_size, color = 'forestgreen', marker = "*", label="-1mm - 1mm", zorder=5)
    geo_df_true[(geo_df_true['true values'] < -0.001)].plot(ax = ax1, markersize = dot_size, color = 'maroon', marker = "*", label="< -1mm", zorder=4)
    # Points for ax2
    geo_df_pred[(geo_df_pred['predicted values'] > 0.001)].plot(ax = ax2, markersize = dot_size, color = 'forestgreen', marker = "*", label="> 1mm", zorder=6)
    geo_df_pred[(geo_df_pred['predicted values'] >= -0.001) & (geo_df_pred['predicted values'] <= 0.001)].plot(ax = ax2, markersize = dot_size, color = 'gold', marker = "*", label="-1mm - 1mm", zorder=5)
    geo_df_pred[(geo_df_pred['predicted values'] < -0.001)].plot(ax = ax2, markersize = dot_size, color = 'maroon', marker = "*", label="< -1mm", zorder=4)

    pyplot.legend(prop={'size': 12})
    pyplot.show()

plot_map()