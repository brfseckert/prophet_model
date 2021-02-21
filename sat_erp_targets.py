#importing packages
import pandas as pd
import numpy as np
import itertools
from fbprophet import Prophet
import time
from multiprocessing import Pool

# disabeling warnings
import warnings
warnings.filterwarnings("ignore")

# defining the folder with our data
target_folder = ''

# function to adjust the format
def adjust_format(data):
 data = str(data)
 data = data.replace(",", "")
 return data


# adjusting the index
def index_ad(data, index):
 data[index] = pd.to_datetime(data[index])
 data = data.set_index(index)
 return data


# function to open and clean the data
def sales_data(file):
 data = pd.read_csv(target_folder + file + '.txt', sep='\t', low_memory=False)
 data = pd.melt(data, id_vars=['ERP','REGION', 'MARKING', 'DISTRICT', 'TERRITORY'], value_name='ship (CT)', var_name='we')
 data['ship (CT)'] = data['ship (CT)'].apply(adjust_format).astype(float).fillna(0)
 data = data[~data['TERRITORY'].astype(str).str.contains("Old")]
 data = data.groupby(['ERP','REGION', 'DISTRICT', 'MARKING', 'TERRITORY', 'we']).sum().reset_index()
 data['TERR_KEY'] = data['REGION'] + ">" + data['MARKING'] + ">" + data['TERRITORY']
 # data['total'] = data[data['we'] >= '2020-01-05'].groupby(['ERP'])['ship (CT)'].transform(sum)
 # data['total'] = data.groupby(['ERP'])['total'].transform(sum).fillna(0)
 # data = data[data['total'] != 0]
 data['we'] = pd.to_datetime(data['we'])
 data = data.set_index('we')
 data = data.sort_values(by='ERP')
 return data['2016-01-03':]


# simplified version of the ctb function
def df_list(df, col='ERP'):
    df_list = []
    key = df[col].unique().tolist()
    for i in key:
        print(i)
        data = df[df[col] == i]
        df_list.append(data)
    return df_list

# def df_list_pool(i):
#     data = df[df[col] == i]
#     df_list.append(data)
# key = df[col].unique().tolist()

# calculatagin the contributions
def ctb(data):
 method = ['yhat']
 for i in method:
     data = data.reset_index
     data['ctb-' + i] = (data[i] /
                         data.groupby(['TERR_KEY', 'index'])[i].transform(sum)).replace(
         [np.inf, -np.inf], [0, 0])
 return data


# simplified version of the fcast
def fcast(df, forecast_steps=52):

    final_data = pd.DataFrame()
    df = pd.DataFrame(df)
    erp = df['ERP'][1]
    terr_key = df['TERR_KEY'][1]
    region = df['REGION'][1]
    district = df['DISTRICT'][1]
    marking = df['MARKING'][1]
    territory = df['TERRITORY'][1]
    df = df['ship (CT)']

    # adjusting the columns for the model
    ship = df
    df = df.reset_index()
    df = df.rename(columns={'we': 'ds', 'ship (CT)': 'y'})
    df = df.fillna(0)

    # adjusting interval of confidence
    my_model = Prophet(interval_width=0.95,
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        uncertainty_samples=300,
                        seasonality_mode='additive'
                        )
    my_model.fit(df)

    # applying the results
    future_dates = my_model.make_future_dataframe(periods=20, freq='W')
    forecast = my_model.predict(future_dates)
    final_data = pd.DataFrame(index_ad(forecast, 'ds')['yhat'])
    # final_data = pd.concat([ship, ifcast1], axis=1)
    final_data['ERP'] = erp
    final_data['TERR_KEY'] = terr_key
    final_data['REGION'] = region
    final_data['MARKING'] = marking
    final_data['DISTRICT'] = district
    final_data['TERRITORY'] = territory

    print(erp)
    return final_data


# simplified version of the execution function
def model_execution(df_list):
     results = []
     for df in df_list:
         df = fcast(df)
         results.append(df)
     return results



if __name__ == '__main__':
    start = time.time()
    data = sales_data('erp_sales')
    df_list = df_list(data, 'ERP')
    pool = Pool()
    data = pool.map(fcast,df_list)
    data = pd.concat(data)
    data.to_csv(r'erp_fcast.csv')
    data = data.reset_index()
    data['ctb-' + 'yhat'] = (data['yhat'] /
                                 data.groupby(['TERRITORY', 'ds'])['yhat'].transform(sum)).replace([np.inf, -np.inf],
                                                                                                    [0, 0])
    data.to_csv(r'erp_fcast.csv')
    end = time.time()
    print(end-start)
    print("SAT Targts were generated sucessfully")