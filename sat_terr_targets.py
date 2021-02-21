# importing packages
import pandas as pd
import numpy as np
from fbprophet import Prophet
import time
from multiprocessing import Pool

# disabeling warnings
import warnings
warnings.filterwarnings("ignore")

# defining the folder with our data
target_folder = r'\\ca.batgen.com\users\MO_USERS_MO\85011301\Documents\felipe_eckert\go_program\target_tool'


# function to adjust the format
def adjust_format(data):
    data = str(data)
    data = data.replace(",", "")
    data = data.replace('2015-01-04', "")
    return data


# adjusting the index
def index_ad(data, index):
    data[index] = pd.to_datetime(data[index])
    data = data.set_index(index)
    return data


def ws_adj(file='erp_sales',terr_file='territory_sales',wc_erp=50334986, ws_wc_erp=50340486, pct=0.07):
    data = pd.read_csv(target_folder + '\\' + file + '.txt', sep='\t', low_memory=False)
    data = data[(data['MARKING'] == 'MB - Manitoba') & (data['TERRITORY'] == '70001 - WHOLESALER')]
    data = pd.melt(data, id_vars=['ERP', 'REGION', 'MARKING', 'DISTRICT', 'TERRITORY'], value_name='ship (CT)',
                   var_name='we')
    data['ship (CT)'] = data['ship (CT)'].apply(adjust_format).astype(float).fillna(0)
    data.set_index('ERP', inplace=True)
    data = data.drop(index=wc_erp).sort_values(by=['we'])
    data = data.reset_index()

    def apply_pct(x):
        if x['ERP'] == ws_wc_erp:
            return x['ship (CT)'] * pct
        else:
            return x['ship (CT)']

    data['ship (CT)'] = data.apply(lambda x: apply_pct(x), axis=1)
    data = data.groupby(['REGION', 'MARKING', 'DISTRICT', 'TERRITORY', 'we']).sum().reset_index()
    terr_data = pd.read_csv(target_folder + '\\' + terr_file + '.txt', sep='\t', low_memory=False)
    terr_data = pd.melt(terr_data, id_vars=['REGION', 'MARKING', 'DISTRICT', 'TERRITORY'], value_name='ship (CT)', var_name='we')

    # adjust the shiping column and assign 1 to the missing values so as to avoid negative values
    terr_data  = terr_data[~((terr_data ['MARKING'] == 'MB - Manitoba')
                              & (terr_data ['TERRITORY'] == '70001 - WHOLESALER'))]
    terr_data = pd.concat([terr_data, data])
    terr_data.to_csv(target_folder + r'\territory_sales_ws.txt',index=False, sep='\t')


# function to open and clean the data
def sales_data(file):
    data = pd.read_csv(target_folder + '\\' + file + '.txt', sep='\t', low_memory=False)
    data['ship (CT)'] = data['ship (CT)'].apply(adjust_format).astype(float).fillna(0)

    # filtering out old territories
    data = data[~data['TERRITORY'].astype(str).str.contains("Old")]

    # filtering out territories that didn't sell anything in 2020
    data = data.groupby(['REGION', 'DISTRICT', 'MARKING', 'TERRITORY', 'we']).sum().reset_index()
    data['TERR_KEY'] = data['REGION'] + ">" + data['MARKING'] + ">" + data['TERRITORY']
    data['total'] = data[data['we'] >= '2020-01-05'].groupby(['TERR_KEY'])['ship (CT)'].transform(sum)
    data['total'] = data.groupby(['TERR_KEY'])['total'].transform(sum).fillna(0)
    data['total_count'] = data.groupby(['TERR_KEY'])['we'].transform('count')
    data = data[data['total'] != 0]
    data = data[data['total_count'] > 1]
    data['we'] = pd.to_datetime(data['we'])
    data = data.set_index('we')
    data['MARKING'] = data.apply(lambda row: 'AB - Alberta' if row['REGION']=='RSM WEST' and row['MARKING']=='MB - Manitoba' else row['MARKING'],axis=1)
    data = data[~(data['TERRITORY'] == 'UNKNWN')]
    return data

# simplified version of the ctb function
def df_list(df, col):
    df_list = []
    key = df[col].unique().tolist()
    for i in key:
        print(i)
        data = df[df[col] == i]
        df_list.append(data)
    return df_list


# calculatagin the contributions
def ctb(data):
    method = ['yhat']
    for i in method:
        data = data.reset_index
        data['ctb-' + i] = (data[i] /
                            data.groupby(['MARKING', 'index'])[i].transform(sum)).replace(
            [np.inf, -np.inf], [0, 0])
    return data


# simplified version of the fcast
def fcast(df, forecast_steps=52):
    final_data = pd.DataFrame()
    print(df['TERR_KEY'][1])
    region = df['REGION'][1]
    district = df['DISTRICT'][1]
    marking = df['MARKING'][1]
    territory = df['TERRITORY'][1]


    # adjusting the columns for the model
    ship = df
    df = df.reset_index()
    df = df.rename(columns={'we': 'ds', 'ship (CT)': 'y'})
    df['y'] = df['y'].fillna(1)

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
    future_dates = my_model.make_future_dataframe(periods=52, freq='W')
    forecast = my_model.predict(future_dates)
    ifcast1 = pd.DataFrame(index_ad(forecast, 'ds'))
    final_data = pd.concat([ship, ifcast1], axis=1)
    final_data['DISTRICT'] = district
    final_data['REGION'] = region
    final_data['MARKING'] = marking
    final_data['TERRITORY'] = territory

    print(marking + territory)
    return final_data

# simplified version of the execution function
def model_execution(df_list):
    results = []
    for df in df_list:
        try:
            df = fcast(df)
            results.append(df)
        except:
            continue
    return results

if __name__ == '__main__':
    start = time.time()
    data = ws_adj()
    data = sales_data('territory_sales_ws')
    df_list = df_list(data, 'TERR_KEY')
    pool = Pool()
    data = pool.map(fcast,df_list)
    # data = model_execution(df_list)
    data = pd.concat(data)
    data = data.reset_index()
    data['ctb-' + 'yhat'] = (data['yhat'] /
                             data.groupby(['MARKING', 'index'])['yhat'].transform(sum)).replace([np.inf, -np.inf],
                                                                                                [0, 0])
    data.to_csv(target_folder + '\\terr_fcast.txt')
    end = time.time()
    print(end-start)
    print("SAT Targts were generated sucessfully")