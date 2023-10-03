import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/PC/Dropbox/eloras_rats/data/all_data_step6columned.csv")
df.columns = df.columns.str.replace(".", "_")

df['accuracy'] = np.where(df['X_correct_entry_'] == 1, "Correct", "Wrong")

df['response_type'] = df['X_group_width_'] + " " + df['X_group_dir_']

df = df.dropna(subset=['X_response_lat_', 'X_correct_entry_', 'first_check_lat'])

summary_data = df.groupby(['response_type', 'accuracy']).agg(mean_response_lat=('X_response_lat_', 'mean'), sd_response_lat=('X_response_lat_', 'std')).reset_index()

p_values = df[df['response_type'].isin(["NARO LM", "WIDE LM", "NARO ML", "WIDE ML"])].groupby('response_type').apply(lambda x: ttest_ind(x['X_response_lat_'][x['accuracy'] == 'Correct'], x['X_response_lat_'][x['accuracy'] == 'Wrong']).pvalue).reset_index(name='p_value')



# Convert R data frame to pandas data frame
df = pd.DataFrame(df)

# Create bar plot with mean values
sns.barplot(x='response_type', y='X.response_lat.', hue='accuracy', data=df, ci='sd', dodge=True)

# Add error bars
sns.barplot(x='response_type', y='X.response_lat.', hue='accuracy', data=df, ci='sd', dodge=True, errwidth=0.8, capsize=0.4)

# Add jittered points
sns.stripplot(x='response_type', y='X.response_lat.', data=df, dodge=True, jitter=0.2, size=3, color='black')

# Set axis labels and title
plt.xlabel('Response type')
plt.ylabel('Response Latency (s)')
plt.legend(title='Accuracy')
plt.title('Response Time Latencies by Subgroup For Normal Durations')

# Set theme
sns.set_style('whitegrid')

# Set legend position
plt.legend(loc='upper center')

# Add significance annotations
p_values = pd.DataFrame(p_values)
for i, row in p_values.iterrows():
    if row['p_value'] < 0.001:
        annotation = '***'
    elif row['p_value'] < 0.01:
        annotation = '**'
    elif row['p_value'] < 0.05:
        annotation = '*'
    else:
        annotation = ''
    plt.annotate(annotation, xy=(row['response_type'], 1.5), xytext=(row['response_type'], 1.5), textcoords='data', ha='center', va='bottom', fontsize=4, color='black')

# Set y-axis limits
plt.ylim(0, 2)

# Show the plot
plt.show()