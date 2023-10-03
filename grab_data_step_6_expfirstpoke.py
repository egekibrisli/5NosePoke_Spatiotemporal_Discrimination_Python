# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:10:27 2023

@author: PC
"""

import pandas as pd
import numpy as np
import os
#=======================================================================================
# UTILS & functions
#=======================================================================================
def make_trial_index(d):
    trial_start_stamp = (d['V5'] == 'StartPoke') & (d['V3'] == 'Entry')
    trial_start_stamp = trial_start_stamp.shift(1).fillna(0).astype(int)
    trial_index = np.cumsum(trial_start_stamp)
    trial_index = trial_index.where(trial_index > 0).fillna(method='ffill').astype(int)
    return trial_index

#=======================================================================================
# Load and preprocess the data
#=======================================================================================
dat = pd.DataFrame()

for ss in range(1, 16):  # sessions number
    dir = f'/Users/PC/Dropbox/eloras_rats/data/step6/S{ss}/'
    ls = [file for file in os.listdir(dir) if file.endswith('.csv')]
    
    for file_name in ls:
        file_path = os.path.join(dir, file_name)
        d = pd.read_csv(file_path, header=None, nrows=4, sep=',', na_values='', keep_default_na=False)
        
        rat_nr = str(d.iloc[0, 3])
        group_info = d.iloc[1, 3].split(' ')
        group_width = group_info[0][:4]
        group_dir = group_info[1][5:7]
        
        d = pd.read_csv(file_path, header=None, skiprows=7, sep=',', na_values='', keep_default_na=False)
        d.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
        d['time_stamp'] = d['V2']
        d['trial_index'] = make_trial_index(d)
        d_len = len(d)
        
        d['session'] = ss
        d['group_width'] = group_width
        d['group_dir'] = group_dir
        d['rat_nr'] = rat_nr
        
        dat = pd.concat([dat, d[:-1]])

dat.reset_index(drop=True, inplace=True)
dat['V6'] = pd.to_numeric(dat['V6'], errors='coerce')

dat.to_csv('/Users/PC/Dropbox/eloras_rats/data/denemestep6/dat_step6expp.csv', index=False)

#=======================================================================================
# GET the trial data + UTILS & functions
#=======================================================================================

dat = pd.read_csv('/Users/PC/Dropbox/eloras_rats/data/denemestep6/dat_step6expp.csv')

def process_trial_data(z):
    tmp1 = z[(z['V5'] == 'StartPoke') & (z['V3'] == 'Entry')]
    tmp2 = z[(z['V5'] == 'StartPort') & (z['V3'] == 'Input')]

    start_light = tmp1['time_stamp'].iloc[0]
    init_entry = tmp2['time_stamp'].iloc[0]

    cond = 0
    if z[(z['V5'] == 'Duration_1s') & (z['V3'] == 'Exit')].shape[0] > 0:
        cond = 1
    elif z[(z['V5'] == 'Duration_2s') & (z['V3'] == 'Exit')].shape[0] > 0:
        cond = 2
    elif z[(z['V5'] == 'Duration_4s') & (z['V3'] == 'Exit')].shape[0] > 0:
        cond = 4

    int_offset = z[(z['V5'].str.startswith('Duration_')) & (z['V3'] == 'Exit')]['time_stamp'].iloc[0]
    first_check = z[(z['V5'].str.contains('NosePoke')) & (z['time_stamp'] > int_offset)]['V4'].iloc[0]

    first_check_lat = z[(z['V5'].str.contains('NosePoke')) & (z['time_stamp'] > int_offset)]['time_stamp'].iloc[0]
    sumNP1 = (z['V5'] == 'NosePoke1').sum()
    sumNP2 = (z['V5'] == 'NosePoke2').sum()
    sumNP3 = (z['V5'] == 'NosePoke3').sum()
    sumNP4 = (z['V5'] == 'NosePoke4').sum()
    sumNP5 = (z['V5'] == 'NosePoke5').sum()
    time_NP1 = z.loc[(z['V5'] == 'NosePoke1') & (z['V3'] == 'Input') & (z['time_stamp'] > 0), 'V2'].values
    time_NP2 = z.loc[(z['V5'] == 'NosePoke2') & (z['V3'] == 'Input') & (z['time_stamp'] > 0), 'V2'].values
    time_NP3 = z.loc[(z['V5'] == 'NosePoke3') & (z['V3'] == 'Input') & (z['time_stamp'] > 0), 'V2'].values
    time_NP4 = z.loc[(z['V5'] == 'NosePoke4') & (z['V3'] == 'Input') & (z['time_stamp'] > 0), 'V2'].values
    time_NP5 = z.loc[(z['V5'] == 'NosePoke5') & (z['V3'] == 'Input') & (z['time_stamp'] > 0), 'V2'].values
    start_lat = z.loc[z[z['V5'].str.startswith('Duration_') & (z['V3'] == 'Entry')].index[-1], 'V2']

    lat_NP1 = min(time_NP1[time_NP1 > start_lat]) - start_lat
    lat_NP2 = min(time_NP2[time_NP2 > start_lat]) - start_lat
    lat_NP3 = min(time_NP3[time_NP3 > start_lat]) - start_lat
    lat_NP4 = min(time_NP4[time_NP4 > start_lat]) - start_lat
    lat_NP5 = min(time_NP5[time_NP5 > start_lat]) - start_lat

    first_poke = None

    lat_NP_values = [lat_NP1, lat_NP2, lat_NP3, lat_NP4, lat_NP5]
    if not any(np.isnan(lat_NP_values)):
        first_poke = lat_NP_values.index(min(lat_NP_values)) + 1

    return [cond, start_light, init_entry, int_offset, first_check, first_check_lat, sumNP1, sumNP2, sumNP3, sumNP4, sumNP5, lat_NP1, lat_NP2, lat_NP3, lat_NP4, lat_NP5, first_poke]

trial_data = dat.groupby(['rat_nr', 'session', 'trial_index', 'group_width', 'group_dir']).apply(process_trial_data).reset_index()
trial_data.columns = ['rat_nr', 'session', 'trial_index', 'group_width', 'group_dir', 'cond', 'start_light', 'init_entry', 'int_offset', 'first_check', 'first_check_lat', 'sumNP1', 'sumNP2', 'sumNP3', 'sumNP4', 'sumNP5', 'lat_NP1', 'lat_NP2', 'lat_NP3', 'lat_NP4', 'lat_NP5', 'first_poke']

# wide group
trial_data['correct_entry'] = 0
cond_1_filter = (trial_data['group_width'] == 'WIDE') & (trial_data['group_dir'] == 'LM') & (trial_data['cond'] == 1) & (trial_data['first_check'] == 3)
cond_2_filter = (trial_data['group_width'] == 'WIDE') & (trial_data['group_dir'] == 'LM') & (trial_data['cond'] == 2) & (trial_data['first_check'] == 5)
cond_4_filter = (trial_data['group_width'] == 'WIDE') & (trial_data['group_dir'] == 'LM') & (trial_data['cond'] == 4) & (trial_data['first_check'] == 1)
cond_5_filter = (trial_data['group_width'] == 'WIDE') & (trial_data['group_dir'] == 'ML') & (trial_data['cond'] == 1) & (trial_data['first_check'] == 3)
cond_6_filter = (trial_data['group_width'] == 'WIDE') & (trial_data['group_dir'] == 'ML') & (trial_data['cond'] == 2) & (trial_data['first_check'] == 1)
cond_8_filter = (trial_data['group_width'] == 'WIDE') & (trial_data['group_dir'] == 'ML') & (trial_data['cond'] == 4) & (trial_data['first_check'] == 5)
cond_9_filter = (trial_data['group_width'] == 'NARO') & (trial_data['group_dir'] == 'LM') & (trial_data['cond'] == 1) & (trial_data['first_check'] == 3)
cond_10_filter = (trial_data['group_width'] == 'NARO') & (trial_data['group_dir'] == 'LM') & (trial_data['cond'] == 2) & (trial_data['first_check'] == 4)
cond_12_filter = (trial_data['group_width'] == 'NARO') & (trial_data['group_dir'] == 'LM') & (trial_data['cond'] == 4) & (trial_data['first_check'] == 2)
cond_13_filter = (trial_data['group_width'] == 'NARO') & (trial_data['group_dir'] == 'ML') & (trial_data['cond'] == 1) & (trial_data['first_check'] == 3)
cond_14_filter = (trial_data['group_width'] == 'NARO') & (trial_data['group_dir'] == 'ML') & (trial_data['cond'] == 2) & (trial_data['first_check'] == 2)
cond_16_filter = (trial_data['group_width'] == 'NARO') & (trial_data['group_dir'] == 'ML') & (trial_data['cond'] == 4) & (trial_data['first_check'] == 4)

trial_data.loc[cond_1_filter | cond_2_filter | cond_4_filter | cond_5_filter | cond_6_filter | cond_8_filter | cond_9_filter | cond_10_filter | cond_12_filter | cond_13_filter | cond_14_filter | cond_16_filter, 'correct_entry'] = 1

trial_data.to_csv('C:/Users/PC/Dropbox/eloras_rats/data/denemestep6/all_data_step6exp.csv', index=False)

#=======================================================================================
# Split single column data into multiple columns
#=======================================================================================
df = pd.read_csv("/Users/PC/Dropbox/eloras_rats/data/denemestep6/all_data_step6exp.csv")
df['data'] = df['data'].str.split()

df = df['data'].apply(pd.Series)
df.columns = ['col' + str(i + 1) for i in range(len(df.columns))]

# Convert columns to their appropriate data types
col_to_int = ['col1', 'col3', 'col4', 'col7']
col_to_float = ['col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24']

df[col_to_int] = df[col_to_int].astype(int)
df[col_to_float] = df[col_to_float].astype(float)

# Write the resulting data frame to a new CSV file
df.to_csv("dnmall_data_step6expcolumned.csv", index=False)

#=======================================================================================
# Read in the data from the CSV file and exclude lat_NP values greater than 20
#=======================================================================================
all_data = pd.read_csv('C:/Users/PC/Dropbox/eloras_rats/data/denemestep6/dnmall_data_step6expcolumned.csv')

# Create four subsets based on the given conditions
subset1 = all_data[(all_data['cond'] == 4) & (all_data['group_width'] == "NARO") & (all_data['group_dir'] == "LM")]
subset2 = all_data[(all_data['cond'] == 4) & (all_data['group_width'] == "NARO") & (all_data['group_dir'] == "ML")]
subset3 = all_data[(all_data['cond'] == 4) & (all_data['group_width'] == "WIDE") & (all_data['group_dir'] == "LM")]
subset4 = all_data[(all_data['cond'] == 4) & (all_data['group_width'] == "WIDE") & (all_data['group_dir'] == "ML")]

# Create SE and ME columns for each subset
subset1['SE'] = np.where(np.isinf(subset1['lat_NP3']), np.nan, subset1['lat_NP3'])
subset1['ME'] = np.where(np.isinf(subset1['lat_NP4']), np.nan, subset1['lat_NP4'])
subset2['SE'] = np.where(np.isinf(subset2['lat_NP3']), np.nan, subset2['lat_NP3'])
subset2['ME'] = np.where(np.isinf(subset2['lat_NP2']), np.nan, subset2['lat_NP2'])
subset3['SE'] = np.where(np.isinf(subset3['lat_NP3']), np.nan, subset3['lat_NP3'])
subset3['ME'] = np.where(np.isinf(subset3['lat_NP5']), np.nan, subset3['lat_NP5'])
subset4['SE'] = np.where(np.isinf(subset4['lat_NP3']), np.nan, subset4['lat_NP3'])
subset4['ME'] = np.where(np.isinf(subset4['lat_NP1']), np.nan, subset4['lat_NP1'])

# For subset 1
subset1['SE'] = np.where(subset1['first_poke'] == 3, 1, 0)
subset1['ME'] = np.where(subset1['first_poke'] == 4, 1, 0)

# For subset 2
subset2['SE'] = np.where(subset2['first_poke'] == 3, 1, 0)
subset2['ME'] = np.where(subset2['first_poke'] == 2, 1, 0)

# For subset 3
subset3['SE'] = np.where(subset3['first_poke'] == 3, 1, 0)
subset3['ME'] = np.where(subset3['first_poke'] == 5, 1, 0)

# For subset 4
subset4['SE'] = np.where(subset4['first_poke'] == 3, 1, 0)
subset4['ME'] = np.where(subset4['first_poke'] == 1, 1, 0)

# Combine the subsets back into one data frame
all_data_filtered = pd.concat([subset1, subset2, subset3, subset4])

# Write the modified data frame to a new CSV file with lat_NP columns
all_data_filtered.to_csv('C:/Users/PC/Dropbox/eloras_rats/data/denemestep6/all_data_step6exp_filtered.csv', index=False)
