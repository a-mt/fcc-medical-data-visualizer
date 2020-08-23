import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = df['weight'] / np.square(df['height']/100)
df['overweight'] = (bmi > 25).astype('uint8')

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['gluc'] = (df['gluc'] != 1).astype('uint8')
df['cholesterol'] = (df['cholesterol'] != 1).astype('uint8')

#-----------------------------------------------------------
# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    columns = [
      'active',
      'alco',
      'cholesterol',
      'gluc',
      'overweight',
      'smoke'
    ]
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=columns)

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.
    df_cat = df_cat.reset_index() \
                .groupby(['variable', 'cardio', 'value']) \
                .agg('count') \
                .rename(columns={'index': 'total'}) \
                .reset_index()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(
        x="variable",
        y="total",
        col="cardio",
        hue="value",
        data=df_cat,
        kind="bar").fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

#-----------------------------------------------------------
# Draw Heat Map
def draw_heat_map():
    # Clean the data
    # - diastolic pressure is higher then systolic
    # - height is less than the 2.5th percentile
    # - height is more than the 97.5th percentile
    # - weight is less then the 2.5th percentile
    # - weight is more than the 97.5th percentile
    df_heat = df[
      (df['ap_lo'] <= df['ap_hi'])
      & (df['height'] >= df['height'].quantile(0.025))
      & (df['height'] <= df['height'].quantile(0.975))
      & (df['weight'] >= df['weight'].quantile(0.025))
      & (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig = plt.figure(figsize=(12,6))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask,
                annot=True, fmt='.1f',
                center=0, vmin=-0.5, vmax=0.5)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
