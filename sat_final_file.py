# importing packages
import pandas as pd
import numpy as np
import time

#disabeling warnings
import warnings
warnings.filterwarnings("ignore")

# defining the folder with our data
target_folder = ''

def adjust_format(data):
    data = str(data)
    data = data.replace(",","")
    return data


# openining and adjusting the territory ctb file
terr_ctb = pd.read_csv(target_folder + r'terr_fcast.csv', sep=',')

# adjusting the terr_ctb data
terr_ctb['total change'].fillna(0, inplace=True)
terr_ctb = pd.melt(terr_ctb,
                   id_vars=['REGION', 'MARKING', 'DISTRICT', 'TERRITORY', 'total change'],
                   value_name='fcast',
                   var_name='we')
# create the lists
we_lst = terr_ctb['we'].unique().tolist()
wk_number = len(we_lst)
print(wk_number)

# temporary week filter
we_lst = we_lst[5:]
print(we_lst)
terr_ctb =terr_ctb[terr_ctb['we'].isin(we_lst)]

# adjusting the territory ctb file
terr_ctb['total change'] = terr_ctb['total change'] / wk_number
terr_ctb['we'] = pd.to_datetime(terr_ctb['we'])
terr_ctb['MARKING'] = terr_ctb['MARKING'].str.strip()
terr_ctb['fcast'] = terr_ctb['fcast'].apply(adjust_format).astype(float)
terr_ctb['fcast'] = terr_ctb['fcast'] + terr_ctb['total change']
terr_ctb['fcast'] = terr_ctb['fcast'].apply(adjust_format).astype(float).fillna(0)
terr_ctb['TOTAL'] = terr_ctb.groupby(['MARKING','REGION','DISTRICT','TERRITORY'])['fcast'].transform(sum).replace([np.inf, -np.inf], [0, 0])
terr_ctb['ctb'] = terr_ctb['TOTAL'] / terr_ctb.groupby(['MARKING'])['fcast'].transform(sum).replace(
    [np.inf, -np.inf], [0, 0])
terr_ctb.to_csv('terr_ctb.csv')

# opening and adjusting the fcast volume
fcast_volume = pd.read_excel(target_folder + r'forecast_volume.xlsx',sheet_name='fcast_volume')

# merging fcast with terr_ctb
terr_sat = terr_ctb.merge(fcast_volume,
                          how='left',
                          left_on=['MARKING','we'],
                          right_on=['Marking','WE'])
terr_sat['SAT_KPC'] = terr_sat['ctb'] * terr_sat['fcast_volume_final']
terr_sat.drop(['total change','fcast','TOTAL','ctb','Marking','WE','week_number','year','fcast_volume','fcast_volume_h2','fcast_volume_final'],axis=1,inplace=True)


# opening and adjusting the erp fcast volume
erp_fcast = pd.read_csv(target_folder + r'erp_fcast.csv')
erp_fcast = erp_fcast[erp_fcast['ds'].isin(we_lst)]

# adjusting the data
erp_fcast['ds'] = pd.to_datetime(erp_fcast['ds'])
marking = erp_fcast['MARKING'].str.split("-",n=1,expand=True)
erp_fcast['MARKING'] = marking[1]
erp_fcast['MARKING'] = erp_fcast['MARKING'].str.strip()

# adjusting the ctb to Manitoba
erp_fcast['MARKING'] = erp_fcast.apply(lambda row: 'Alberta' \
                                       if row['REGION']=='RSM WEST' and row['MARKING']=='Manitoba'\
                                       else row['MARKING'],axis=1)

# merging SAT with ERP fcast
erp_sat = erp_fcast.merge(terr_sat, how='inner',
                          left_on=['MARKING','REGION','DISTRICT','TERRITORY','ds'],
                          right_on=['MARKING','REGION','DISTRICT','TERRITORY','we'])

# adjusting the ctb to each territory
erp_sat['ctb-' + 'yhat'] = (erp_sat['yhat'] /
                         erp_sat.groupby(['MARKING','REGION','DISTRICT','TERRITORY', 'ds'])['yhat'].transform(sum)).replace([np.inf, -np.inf],
                                                                                           [0, 0])
# creating the new columns
erp_sat['Sales_Target_Qty'] = erp_sat['ctb-yhat']*erp_sat['SAT_KPC']*5
erp_sat['Measurement_Unit_Code'] = 'CAR'
erp_sat['House_Code'] = ''
erp_sat['Material_Subgroup_Code'] = ''
erp_sat = erp_sat.rename(columns={'ds':'Week_End_Date','ERP':'Customer_Nbr'})

# adjusting WS customers
erp_list = [50334561, 50334477, 50334986]
pro = ['Alberta', 'Saskatchewan', 'Manitoba']

# WC negative list
terr_sat = erp_sat[(erp_sat['TERRITORY'] != '70001 - WHOLESALER') | ~(erp_sat['MARKING'].isin(pro))]
ws_sat = erp_sat[(erp_sat['TERRITORY'] == '70001 - WHOLESALER') & (erp_sat['MARKING'].isin(pro))]


# adjusting ws ero customers
def sales(data):
    if data['Customer_Nbr'] in erp_list:
        return 0
    else:
        return data['yhat']

def final_sat(data):
    if data['Customer_Nbr'] in erp_list:
        return data['SAT_KPC'] * 5 - data['total']
    else:
        return data['yhat']

# applying the functions into the datset
ws_sat['Sales_Target_Qty'] = ws_sat.apply(sales, axis=1)
ws_sat['total'] = ws_sat.groupby(['Week_End_Date', 'TERR_KEY'])['Sales_Target_Qty'].transform(sum)
ws_sat['Sales_Target_Qty'] = ws_sat.apply(final_sat, axis=1)
ws_sat.drop(['total'], axis=1, inplace=True)

# joing the datasets
erp_sat = pd.concat([terr_sat, ws_sat])

erp_sat = erp_sat[['Customer_Nbr','Week_End_Date','House_Code','Material_Subgroup_Code','Measurement_Unit_Code','Sales_Target_Qty','yhat','REGION','MARKING','DISTRICT','TERRITORY']]

# saving the file
erp_sat.to_csv('sat_final/' +'SAT_Targets_C2_C3_2021.csv',index=False)