
import pandas as pd
import numpy as np
import math
from scipy.stats import ttest_rel, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from the CSV file
data = pd.read_csv("/Users/PC/Dropbox/eloras_rats/data/all_data_step6exp_filtered.csv")
# Filter the data for the four groups and SE/ME values smaller than or equal to 15

naro_lm = data[(data['group_width'] == "NARO") & (data['group_dir'] == "LM") & (data['cond'] == 4)]
naro_ml = data[(data['group_width'] == "NARO") & (data['group_dir'] == "ML") & (data['cond'] == 4)]
wide_lm = data[(data['group_width'] == "WIDE") & (data['group_dir'] == "LM") & (data['cond'] == 4)]
wide_ml = data[(data['group_width'] == "WIDE") & (data['group_dir'] == "ML") & (data['cond'] == 4)]
# Calculate the mean and standard error (SE) of SE and ME for each group

mean_SE = [np.mean(group['SE'].dropna()) for group in [naro_lm, naro_ml, wide_lm, wide_ml]]
mean_ME = [np.mean(group['ME'].dropna()) for group in [naro_lm, naro_ml, wide_lm, wide_ml]]
se_SE = [np.std(group['SE'].dropna()) / math.sqrt(len(group['SE'])) for group in [naro_lm, naro_ml, wide_lm, wide_ml]]
se_ME = [np.std(group['ME'].dropna()) / math.sqrt(len(group['ME'])) for group in [naro_lm, naro_ml, wide_lm, wide_ml]]

# Perform statistical tests for significance between SE and ME values in each group

p_values = []
for group in [naro_lm, naro_ml, wide_lm, wide_ml]:
    if len(group['SE']) == len(group['ME']):
        t_test = ttest_rel(group['SE'], group['ME'])
        p_values.append(t_test.pvalue)
    else:
        wilcox_test = wilcoxon(group['SE'], group['ME'])
        p_values.append(wilcox_test.pvalue)
# Create a data frame for mean values, standard errors, and p-values

df = pd.DataFrame({
    'Groups': ["NARO LM", "NARO ML", "WIDE LM", "WIDE ML"],
    'SE_mean': mean_SE,
    'ME_mean': mean_ME,
    'SE_se': se_SE,
    'ME_se': se_ME,
    'p_value': p_values
})
# Create a function to add significance asterisks based on p-values

def add_significance(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''
# Add significance levels to the data frame

df['significance'] = df['p_value'].apply(add_significance)
# Create a long format data frame for plotting

df_long = pd.melt(df, id_vars=['Groups', 'SE_se', 'ME_se', 'p_value', 'significance'], var_name='variable', value_name='value')
# Define custom colors
colors = [""]
# Define custom theme

custom_theme = sns.set_theme(style="whitegrid")
custom_theme.set(title="ME and SE Latencies by Groups For Normal Durations", xlabel="Latencies From Duration Start (s)", ylabel="")
custom_theme.set(font_scale=1.2)
# Plot the graph with standard error in error bars

plt.figure(figsize=(10, 6))
sns.barplot(data=df_long, x='value', y='Groups', hue='variable', alpha=0.8)
sns.errorbar(data=df_long, x='value', y='Groups', xerr=df_long.apply(lambda row: row['SE_se'] if row['variable'] == 'SE_mean' else row['ME_se'], axis=1), linewidth=0.5, capsize=3, fmt='none', color='black')
plt.axvline(x=1, linestyle='dashed', color='black')
plt.axvline(x=2, linestyle='dashed', color='black')
plt.legend(title='Variable')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.text(x=df_long.loc[df_long['variable'] == 'ME_mean', 'value'], y=df_long.loc[df_long['variable'] == 'ME_mean', 'Groups'], s=df_long.loc[df_long['variable'] == 'ME_mean', 'significance'], ha='center', va='bottom', fontsize=10)
plt.show()