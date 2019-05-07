import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

url='http://users.du.se/~h16wilwi/gik258/data/ANN-interpolerad.xlsx'
dataset = pd.read_excel(url, skiprows=3)
print(dataset)

dataset = dataset.drop(['pnt_id', 'pnt_lat', 'pnt_lon', 'pnt_demheight', 'pnt_height', 'pnt_quality', 'pnt_linear'], axis=1)
dataset

dataset.set_index('index', inplace=True)
dataset = dataset.drop(['TYta_mean', 'Daggp_mean', 'Lufu_mean', 'TYtaDaggp_mean'])

dataset_GP = dataset.iloc[:1159, :]
dataset_W = dataset.iloc[1159:, :]

print(dataset_GP)
dataset_W

dataset_W = dataset_W.transpose()
dataset_GP = dataset_GP.transpose()
dataset_W
dataset_GP

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

prediction_list = list()

for index in range(len(2)):
        df = dataset_GP.iloc[:, index]
        #df = pd.concat([df,dataset_W], sort=False)
        df = pd.concat([df, dataset_W], axis=1)
        print('index: {}'.format(index))

        prediction = loaded_model.predict_proba(df)
        prediction_list.append(prediction)
prediction_list
        


