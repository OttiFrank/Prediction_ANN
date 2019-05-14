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

url='http://users.du.se/~h16wilwi/gik258/data/ANN-interpolerad.xlsx'
dataset = pd.read_excel(url, skiprows=3)

#lat_long_only = dataset[['pnt_lat','pnt_lon']]
lat_long_only = dataset.iloc[:1159, 2:4]
lat_long_only
dataset = dataset.drop(['pnt_id', 'pnt_lat', 'pnt_lon', 'pnt_demheight', 'pnt_height', 'pnt_quality', 'pnt_linear'], axis=1)

dataset.set_index('index', inplace=True)
dataset = dataset.drop(['Daggp_mean', 'TYtaDaggp_mean'])

# Ground data
dataset_GP = dataset.iloc[:1159, 927:]
# Weather data
dataset_W = dataset.iloc[1159:, 927:]
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
# For each data point
for index  in range(5):        
        df = dataset_GP.iloc[:, i]
        df = pd.concat([df, dataset_W], axis=1)
        df
        print('index: {}'.format(i))
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
        prediction = loaded_model.predict_proba(val)
        prediction_list.append(prediction)
        i += 1
        
prediction_list
pred_list = []
summary = list()
len(prediction_list[0])
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



df = pd.read_excel('predicted.xlsx', index_col=0)
df
df_true, df_pred = df, df
df_true = df_true.drop('predicted values', axis=1)
df_pred = df_pred.drop('true values', axis=1)

absolute_values = list()
counter= 0;
for i in range(len(df_pred)): 
    pred, true = df_pred.iloc[i,2], df_true.iloc[i,2]
    df_pred.iloc[0,2]
    df_true.iloc[0,2]
    if pred > true:
        precentage = (+((pred-true)/true))   
        counter += 1
    else:
        percentage = (+((true-pred)/true))   
    absolute_values.append(percentage)
counter
absolute_values

my_list = pd.Series(absolute_values)
my_list = my_list.replace(-np.inf, 0)
my_list.hist()
pyplot.show()

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
    dot_size = 6

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
    geo_df_true[(geo_df_true['true values'] >= -0.001) & (geo_df_true['true values'] < 0)].plot(ax = ax1, markersize = dot_size, color = 'maroon', marker = "*", label="-1mm < 0mm", zorder=6)
    geo_df_true[(geo_df_true['true values'] >= 0) & (geo_df_true['true values'] <= 0.001)].plot(ax = ax1, markersize = dot_size, color = 'gold', marker = "*", label="0mm - 1mm", zorder=6)
    # Points for ax2
    geo_df_pred[(geo_df_pred['predicted values'] >= -0.001) & (geo_df_pred['predicted values'] < 0)].plot(ax = ax2, markersize = dot_size, color = 'maroon', marker = "*", label="-1mm < 0mm", zorder=6)
    geo_df_pred[(geo_df_pred['predicted values'] >= 0) & (geo_df_pred['predicted values'] <= 0.001)].plot(ax = ax2, markersize = dot_size, color = 'gold', marker = "*", label="0mm - 1mm", zorder=6)
    '''
    geo_df_pred[(geo_df_true['true values'] - geo_df_pred['predicted values'] <= -0.001)].plot(
                    ax = ax2, 
                    markersize = dot_size, 
                    color = 'maroon', 
                    marker = "o", 
                    label = "Felmarginal på 1.5mm (+)", 
                    zorder=6)
    print("---- Påbörjar uträkningarna andra plotten -----")
   
    geo_df_pred[(geo_df_true['true values'] - geo_df_pred['predicted values'] > -0.001) & 
                (geo_df_true['true values'] - geo_df_pred['predicted values'] < -0.0002)].plot(
                    ax = ax2, 
                    markersize = dot_size, 
                    color = 'orangered', 
                    marker = "o", 
                    label = "Felmarginal på 0.2-1mm (+)", 
                    zorder=5)
    geo_df_pred[(geo_df_true['true values'] - geo_df_pred['predicted values'] >= -0.0001) & 
                (geo_df_true['true values'] - geo_df_pred['predicted values'] <= 0.0001)].plot(   ax = ax2, 
                    markersize = dot_size, 
                    color = 'gold', 
                    marker = "o", 
                    label = "Ingen förändring", 
                    zorder=4)
    geo_df_pred[(geo_df_true['true values'] - geo_df_pred['predicted values'] > 0.001) & 
                (geo_df_true['true values'] - geo_df_pred['predicted values'] <= 0.0002)].plot(
                    ax = ax2, 
                    markersize = dot_size, 
                    color = 'forestgreen', 
                    marker = "o", 
                    label = "Felmarginal på 0.2-1mm (-)", 
                    zorder=5)
    geo_df_pred[(geo_df_true['true values'] - geo_df_pred['predicted values'] > 0.001)].plot(
                    ax = ax2, 
                    markersize = dot_size, 
                    color = 'midnightblue', 
                    marker = "o", 
                    label = "Felmarginal under 0.1mm (-)", 
                    zorder=6)
    '''
    #geo_df[(geo_df['predicted values'] > -0.02) & (geo_df['predicted values'] < 0)].plot(ax = ax2, markersize = dot_size, color = 'yellow', marker = "o", label = "Mindre sättning", zorder=4)
    #geo_df[(geo_df['predicted values'] > -0.09) & (geo_df['predicted values'] <= -0.02)].plot(ax = ax2, markersize = dot_size, color = 'orange', marker = "o", label = "Medel sättning", zorder=5)
    #geo_df[(geo_df['predicted values'] <= -0.021)].plot(ax = ax2, markersize = dot_size, color = 'red', marker = "^", label = "Högst sättning", zorder=6)

    pyplot.legend(prop={'size': 12})
    pyplot.show()

plot_map()


    


