import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.metrics import mean_squared_error

def load_media():
    media = pd.read_csv('data/MMM_test_data.csv', )
    media["start_of_week"] = pd.to_datetime(media["start_of_week"], format="%d-%m-%y")
    # 24-07-22
    media.sort_values(by="start_of_week", inplace=True)
    media.reset_index(drop=True, inplace=True)
    return media

def plot_media(media):
    return media.plot(x='start_of_week', title='MMM test data', legend=True)

def plot_predictions(media, predictions, title):
    observed = media['revenue'] / media['revenue'].mean()
    filename = title.replace(' ', '_')
    mse_str = ""
    for label, pred in predictions.items():
        mse = mean_squared_error(observed, pred)
        mse_str += f"  {label}={mse:.2f}"
    
    
    title = f'{title} (MSE: {mse_str})'
    plt.figure(figsize=(10, 6))
    observed = media['revenue'] / media['revenue'].mean()
    plt.plot(observed, label='Observed')

    for label, pred in predictions.items():
        plt.plot(pred, label=label)

    dates = media['start_of_week']
    plt.xticks(np.arange(0, len(dates), 10), dates.dt.date[::10], rotation=45)
    plt.title(title)    
    plt.legend()

    plt.savefig(f'report/img/{filename}.png')
    plt.show()

def calc_channel_contribution(summary, media):
    mean_vals = summary['mean']
    contribution = mean_vals[mean_vals.index.str.contains('contribution_spend_channel_')]

    def extract_channel_number(col_name):
        match = re.search(r'_(\d+)\[', col_name)
        if match:
            return match.group(1)
        return None


    channel_numbers = [extract_channel_number(i) for i in contribution.index]
    values = pd.DataFrame(index=contribution.index, data=contribution.values, columns=['contribution (%)'])
    values['channel'] = channel_numbers

    # group by channel and sum the contributions
    channel_contributions = values.groupby('channel').mean()
    channel_contributions = channel_contributions.sort_values(by='contribution (%)', ascending=False)
    channel_contributions['contribution (abs)'] = channel_contributions['contribution (%)'] * media['revenue'].sum()
    channel_contributions['spending'] = channel_contributions.index.map(lambda i: media[f'spend_channel_{i}'].sum())

    # calculate ROI 

    channel_contributions['ROI'] = channel_contributions['contribution (abs)'] / channel_contributions['spending']

    return channel_contributions


def plot_roi_contribution(df, title='Channel Contribution and ROI'):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    int_index = df.index.astype(int)

    contribution_color = '#87CEFA'  # Light Sky Blue
    roi_color = '#90EE90'

    # Bar chart for absolute contributions
    bar_width = 0.24
    width = 0.2
    ax1.bar(int_index - bar_width/2, df['contribution (%)'], width=width, color=contribution_color, align='center', label='Contribution (%)')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Contribution (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for ROI
    ax2 = ax1.twinx()
    ax2.bar(int_index + bar_width/2, df['ROI'], width=width, color=roi_color, align='center', label='ROI')
    ax2.set_ylabel('ROI', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Title and layout adjustments
    plt.title(title)
    fig.tight_layout()

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # save the plot
    filename = title.replace(' ', '_')
    plt.savefig(f'report/img/{filename}.png')
    # Show plot
    plt.show()