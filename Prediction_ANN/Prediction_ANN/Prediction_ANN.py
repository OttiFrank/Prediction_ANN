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
from sklearn.metrics import mean_squared_error
from math import sqrt

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

# Extract difference between predicted and true values
def get_difference():
    df = pd.read_excel('predicted.xlsx', index_col=0)
    df = df.iloc[:, 2:4]
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    distance = list()
    for i in range(len(df)):
        df['true values'][i], df['predicted values'][i] = (df['true values'][i] * 1000), (df['predicted values'][i] * 1000)
        distance_value = abs(df['true values'][i] - df['predicted values'][i])
        distance.append(distance_value)
    df.columns = ['riktiga värden', 'predikterade värden']
    
    out = pd.cut(distance, bins=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.4, max(distance)], include_lowest=True)
    ax = out.value_counts().plot.bar(rot=0, color="b", figsize=(15,9))
    pyplot.title('Differens mellan predikterade och riktiga värden', fontsize=18)
    pyplot.xlabel('Differens mätt i millimeter', fontsize=14)
    pyplot.ylabel('Antal värden', fontsize=14)
    pyplot.show()

# Extract predicted and true values
def get_predicted_and_true_values():
    df = pd.read_excel('predicted.xlsx', index_col=0)
    df = df.iloc[:, 2:4]
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    for i in range(len(df)):
        df['true values'][i], df['predicted values'][i] = (df['true values'][i] * 1000), (df['predicted values'][i] * 1000)
    df.columns = ['riktiga värden', 'predikterade värden']
    df1 = df.head(15)
    df1.plot(kind='bar', figsize=(18,10))
    pyplot.grid(which='major', linestyle='-', linewidth='0.8', color='grey')
    pyplot.grid(which='minor', linestyle=':', linewidth='0.8', color='black')
    pyplot.yticks(np.arange(0,-32, step = -2), fontsize=12)
    pyplot.xticks(rotation=0, fontsize=12)
    pyplot.title('Predikterade värden mot riktiga värden', fontsize=18)
    pyplot.xlabel('InSAR-mätningar', fontsize= 16)
    pyplot.ylabel('Millimeter (-)', fontsize=16)
    pyplot.legend(fontsize=12)
    pyplot.show()

# Creates a file with the last predicted and true value for all InSAR-measurments
def create_prediction_file():
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
    # Weather data
    dataset_W = dataset.iloc[1159:, :]
    # Transpose dataset
    dataset_W = dataset_W.transpose()
    #dataset_GP = dataset_GP.transpose()

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
    RMSE_list = list()
    # from column 927 (points)
    i = 927

    # For each data point
    for index  in range(1159):        
            df = dataset_GP.iloc[index, :]
            df = pd.concat([df, dataset_W], axis=1)
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

            # invert scaling for actual
            test_y = val_test.reshape((len(val_test), 1))
            inv_y = np.concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,0]

            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
            RMSE_list.append(rmse)
            i += 1

    mean = np.mean(RMSE_list)

    pred_list = pd.DataFrame([prediction_list])
    pred_list = pred_list.transpose()
    pred_list.reset_index(level=0, inplace=True)
    pred_list.set_index('index')
    true_values = pd.DataFrame([dataset.iloc[:1159,-1]])
    true_values = true_values.transpose()

    df = pd.concat([true_values, pred_list], axis=1, sort=False)
    result = pd.concat([lat_long_only, df], axis=1, sort=False)
    result = result.drop(['index'], axis=1)
    result.columns = ["lat", "lon", 'true values', 'predicted values']

    # Writes an Excel file
    writer = ExcelWriter('predicted.xlsx', engine='xlsxwriter')
    writer.book.use_zip64()
    result.to_excel(writer, sheet_name="Blad1")
    writer.save()
    print(result)    

# Plot ground level change + rain sum
def plot_ground_level_change():
    url='http://users.du.se/~h16wilwi/gik258/data/ANN-interpolerad.xlsx'
    dataset = pd.read_excel(url, skiprows=3)

    #lat_long_only = dataset[['pnt_lat','pnt_lon']]
    lat_long_only = dataset.iloc[:1159, 2:4]
    dataset = dataset.drop(['pnt_id', 'pnt_lat', 'pnt_lon', 'pnt_demheight', 'pnt_height', 'pnt_quality', 'pnt_linear'], axis=1)

    dataset.set_index('index', inplace=True)
    dataset = dataset.drop(['Daggp_mean', 'TYtaDaggp_mean'])

    # Ground data
    dataset_GP = dataset.iloc[199, :]
    # Weather data
    dataset_W = dataset.iloc[1159:, :]
    # Transpose dataset
    dataset_W = dataset_W.transpose()
    dataset_GP = dataset_GP.transpose()

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


    # For one data point      
    df = dataset_GP
    df = pd.concat([df, dataset_W], axis=1, sort=False)
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
    #last_predicted = inv_yhat[-1]
    # invert scaling for actual
    test_y = val_test.reshape((len(val_test), 1))
    inv_y = np.concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # Change to millimeter
    for i in range(len(inv_yhat)):
        inv_yhat[i] = inv_yhat[i] * 1000
        inv_y[i] = inv_y[i] * 1000

    pred_list = pd.DataFrame([inv_yhat, inv_y])
    pred_list = pred_list.transpose()
    pred_list.reset_index(level=0, inplace=True)
    pred_list = pred_list.set_index('index')
    pred_list.columns = ['predicted values', 'actual values']

    # Extract weather conditions 
    dataset_W = dataset_W.iloc[:, :5]
    #rain_values = pd.DataFrame(dataset_W['Rain_mm_sum'])

    # Reset index to numbers
    dataset_W.reset_index(level=0, inplace=True)
    df = pd.merge(dataset_W, pred_list, left_index=True, right_index=True)
    df = df.set_index('index')

    pyplot.plot(df['predicted values'], color='blue', label='Predikterade värden')
    pyplot.plot(df['actual values'], color='orange', label='Riktiga värden')
    pyplot.plot(df['TLuft_mean'], color='c', label='Temperatur luft')
    pyplot.plot(df['TYta_mean'], color='m', label='Temperatur yta')
    pyplot.plot(df['Lufu_mean'], color='y', label='Luftfuktighet')
    pyplot.plot(df['Rain_mm_sum'], color='b', label='Regnmängd')
    pyplot.plot(df['Snow_mm_sum'], color='g', label='Snömängd')
    pyplot.xlabel('Tid')
    pyplot.title('Marksättning och väderförhållande under en treårsperiod')
    pyplot.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    pyplot.show()

# Plot difference on map
def plot_difference():
    # Get difference
    df = pd.read_excel('predicted.xlsx', index_col=0)
    df = df.iloc[:, 2:4]
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    difference_list = list()
    for i in range(len(df)):
        df['true values'][i], df['predicted values'][i] = (df['true values'][i] * 1000), (df['predicted values'][i] * 1000)
        distance_value = abs(df['true values'][i] - df['predicted values'][i])
        difference_list.append(distance_value)
    df.columns = ['true values', 'predicted values']
    # Extract true and predicted values
    df = pd.read_excel('predicted.xlsx', index_col=0)
    df_true, df_pred = df, df
    df_true = df_true.drop('predicted values', axis=1)
    df_pred = df_pred.drop('true values', axis=1)
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    difference_list = pd.DataFrame([difference_list])
    difference_list = difference_list.transpose()
    difference_list.columns = ['difference']
    df_pred = pd.merge(df_pred, difference_list, left_index=True, right_index=True)
    # concat multible shapefiles into one gpd df
    folder = Path("shp")
    print("---- Letar efter och concat av shape-filer ---- ")
    gdf = pd.concat([
        gpd.read_file(shp)
        for shp in folder.glob("*.shp")
    ], sort=False).pipe(gpd.GeoDataFrame)

    print("--- Startar skapandet av GeoDataFrame -----")
    geo_df_pred = gpd.GeoDataFrame(df_pred,
                              crs= 'merc',
                              geometry = geometry)
    print(" ----------------------------------------- ")
    # Limit map size
    xlim = ([12.030, 12.0475])
    ylim = ([57.615, 57.640])
    
    # dot size
    dot_size = 4

    fig, ax1 = pyplot.subplots(1, sharey=True, figsize=(15,15))

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title('Differens mellan predikterade och riktiga värden', fontsize='xx-large')
    print("--- Plottar ut kartan ----")
    gdf.plot(ax = ax1, alpha=0.8, zorder=0)

    print("---- Påbörjar uträkningarna av plott -----")
    # Points for ax1
    geo_df_pred[(geo_df_pred['difference'] > 1)].plot(ax = ax1, markersize =8 , color = 'r', marker = "*", label="> 1mm", zorder=6)
    geo_df_pred[(geo_df_pred['difference'] <= 1)].plot(ax = ax1, markersize = dot_size, color = 'g', marker = "*", label="< 1mm", zorder=5)

    pyplot.legend(prop={'size': 12})
    pyplot.show()


if __name__ == "__main__":
    create_prediction_file()
    get_difference()
    get_predicted_and_true_values()
    plot_ground_level_change()
    plot_difference()