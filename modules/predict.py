import os
import os.path
import dill
import pandas as pd
from datetime import datetime
import json
import logging


path = os.environ.get('PROJECT_PATH', '..')
def predict():
    model = sorted(os.listdir(f'{path}/data/models'))

    with open(f'{path}/data/models/{model[-1]}', 'rb') as file:
        model = dill.load(file)

    df_predict = pd.DataFrame(columns= ['id', 'predict'])
    files_list = os.listdir(f'{path}/data/test')

    for filename in files_list:
        with open(f'{path}/data/test/{filename}', 'r') as file:
            form = json.load(file)
        data = pd.DataFrame([form])
        prediction = model.predict(data)

        dict_pred = {'id': data['id'].values[0], 'predict': prediction[0]}
        df = pd.DataFrame([dict_pred])
        df_predict = pd.concat([df, df_predict], ignore_index=True)


    preds_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df_predict.to_csv(preds_filename, index=False)
    logging.info(f'Predictions are saved as {preds_filename}')
if __name__ == '__main__':
    predict()
