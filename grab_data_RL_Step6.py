import os
import pandas as pd
import numpy as np
import re

def make_trial_index(d):
    trial_start_stamp = np.where((d['V5'] == 'StartPoke') & (d['V3'] == 'Entry'))[0] - 1
    trial_start_stamp = trial_start_stamp[~np.isnan(trial_start_stamp)]
    time_stamps = np.concatenate(([1], trial_start_stamp, len(d['V5'])))
    trial_index = np.array([], dtype=int)
    
    for i in range(1, len(time_stamps)):
        trial_index = np.concatenate((trial_index, np.repeat(i - 1, time_stamps[i] - time_stamps[i - 1])))
    
    trial_index = trial_index[trial_index != 1]
    d['trial_index'] = trial_index
    d = d[d['trial_index'] != 1]
    
    return d

dat = pd.DataFrame(columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'time_stamp', 'trial_index', 'session', 'group_width', 'group_dir', 'rat_nr'])

for ss in range(1, 16):  # sessions number
    dir_path = f'C:/Users/PC/Dropbox/eloras_rats/data/step6/S{ss}/'
    file_list = os.listdir(dir_path)
    
    csv_list = [filename for filename in file_list if filename.endswith('.csv')]
    
    for csv_file in csv_list:
        d = pd.read_csv(os.path.join(dir_path, csv_file), header=None, nrows=4, sep=',', fillna=True)
        rat_nr = str(d[3][0])
        group_width = str(d[4][1])[:4]
        group_dir = str(d[4][1])[5:7]
        
        d = pd.read_csv(os.path.join(dir_path, csv_file), header=None, skiprows=7, sep=',', fillna=True)
        d = d[:-4]  # Remove the last 4 rows which are trash
        d['time_stamp'] = d['V2']  # + d['V3'] / 10^3
        d = make_trial_index(d)
        d_len = len(d)
        d = d.iloc[:-1]
        
        session_col = pd.Series([ss] * d_len)
        group_width_col = pd.Series([group_width] * d_len)
        group_dir_col = pd.Series([group_dir] * d_len)
        rat_nr_col = pd.Series([rat_nr] * d_len)
        
        d = pd.concat([d, session_col, group_width_col, group_dir_col, rat_nr_col], axis=1)
        dat = pd.concat([dat, d])

dat.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'time_stamp', 'trial_index', 'session', 'group_width', 'group_dir', 'rat_nr']

dat['V6'] = pd.to_numeric(dat['V6'])

def extract_durations(dat):
    duration_stamps = []
    temp_durations = []
    
    for i in range(len(dat)):
        if re.search("Duration_", dat.iloc[i]["V5"]) and dat.iloc[i]["V3"] == "Exit":
            temp_durations.append(dat.iloc[i]["V2"])
            
            for j in range(i + 1, min(i + 8, len(dat))):
                if re.search("Duration_", dat.iloc[j]["V5"]) and dat.iloc[i]["V3"] == "Exit":
                    temp_durations = []
                    temp_durations.append(dat.iloc[j]["V2"])
                    break
                elif j == min(i + 7, len(dat)):
                    duration_stamps.append(temp_durations)
                    temp_durations = []
    
    return duration_stamps

def response_latency(response_nose, duration_stamps):
    true_duration_stamps = [x[1:] if len(x) == 2 else x for x in duration_stamps]
    unlisted = [x if isinstance(x, list) and len(x) == 1 else [x] for x in true_duration_stamps]
    
    response_latency = [response - unlisted_value for response, unlisted_value in zip(response_nose, unlisted)]
    
    return response_latency

all_data = dat.groupby(['rat_nr', 'session', 'trial_index', 'group_width', 'group_dir']).apply(lambda z: 
    pd.Series({
        'cond': z['V5'].eq('Duration_1s').sum() * 1 + z['V5'].eq('Duration_2s').sum() * 2 + z['V5'].eq('Duration_4s').sum() * 4,
        'start_light': z[(z['V5'] == 'StartPoke') & (z['V3'] == 'Entry')]['time_stamp'].values[0],
        'init_entry': z[(z['V5'] == 'StartPort') & (z['V3'] == 'Input')]['time_stamp'].values[0],
        'int_offset': z[(z['V5'].str.contains('Duration_')) & (z['V3'] == 'Exit')]['time_stamp'].values[0],
        'first_check': z[(z['V5'].str.contains('NosePoke')) & (z['time_stamp'] > z[(z['V5'].str.contains('Duration_')) & (z['V3'] == 'Exit')]['time_stamp'].values[0])]['V4'].values[0],
        'first_check_lat': z[(z['V5'].str.contains('NosePoke')) & (z['time_stamp'] > z[(z['V5'].str.contains('Duration_')) & (z['V3'] == 'Exit')]['time_stamp'].values[0])]['time_stamp'].values[0],
        'response_latency': response_latency(z[(z['V5'].str.contains('NosePokeChoice_')) & (z['V3'] == 'Exit')]['time_stamp'].values, extract_durations(z))
    })
)

all_data = all_data.reset_index()

def calculate_correct_entry(row):
    if (row['group_width'] == 'WIDE' and row['group_dir'] == 'LM' and row['cond'] == 1 and row['first_check'] == 3) or \
       (row['group_width'] == 'WIDE' and row['group_dir'] == 'LM' and row['cond'] == 2 and row['first_check'] == 5) or \
       (row['group_width'] == 'WIDE' and row['group_dir'] == 'LM' and row['cond'] == 4 and row['first_check'] == 1) or \
       (row['group_width'] == 'WIDE' and row['group_dir'] == 'ML' and row['cond'] == 1 and row['first_check'] == 3) or \
       (row['group_width'] == 'WIDE' and row['group_dir'] == 'ML' and row['cond'] == 2 and row['first_check'] == 1) or \
       (row['group_width'] == 'WIDE' and row['group_dir'] == 'ML' and row['cond'] == 4 and row['first_check'] == 5) or \
       (row['group_width'] == 'NARO' and row['group_dir'] == 'LM' and row['cond'] == 1 and row['first_check'] == 3) or \
       (row['group_width'] == 'NARO' and row['group_dir'] == 'LM' and row['cond'] == 2 and row['first_check'] == 4) or \
       (row['group_width'] == 'NARO' and row['group_dir'] == 'LM' and row['cond'] == 4 and row['first_check'] == 2) or \
       (row['group_width'] == 'NARO' and row['group_dir'] == 'ML' and row['cond'] == 1 and row['first_check'] == 3) or \
       (row['group_width'] == 'NARO' and row['group_dir'] == 'ML' and row['cond'] == 2 and row['first_check'] == 2) or \
       (row['group_width'] == 'NARO' and row['group_dir'] == 'ML' and row['cond'] == 4 and row['first_check'] == 4):
        return 1
    else:
        return 0

all_data['correct_entry'] = all_data.apply(calculate_correct_entry, axis=1)

all_data.to_csv('C:/Users/PC/Dropbox/eloras_rats/data/all_data_step6.csv', index=False)
