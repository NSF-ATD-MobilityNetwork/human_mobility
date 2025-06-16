import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import pickle
from plotly import graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline



def create_dfs(weekly_patterns_folder):
    list_dfs = []
    files = sorted(os.listdir(weekly_patterns_folder))
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(weekly_patterns_folder, file), dtype=str)
            df = df.rename(columns={'poi_cbg':'poi_id', 'visitor_cbg':'visitor_id'})
            df['naics_code'] = df['naics_code'].str[:-2]
            df['count'] = df['count'].astype(int)
            list_dfs.append(df)
    return list_dfs

def filter_dfs(df_list, column, starts_with):
    new_list = []
    for df in df_list:
        if starts_with == '':
            new_list.append(df)
        else:
            df = df[df[column].astype(str).str.startswith(starts_with)]
            new_list.append(df)
    return new_list

def convert_to_tracts(df_list):
    df_list_new = []
    for df in df_list:
        df_new = df.copy()
        df_new['poi_id'] = df_new['poi_id'].str[5:-1]
        df_new['visitor_id'] = df_new['visitor_id'].str[5:-1]
        df_new = df_new.groupby(['naics_code', 'poi_id', 'visitor_id'], as_index=False)['count'].sum()
        df_list_new.append(df_new)
    return df_list_new

def get_naics_df(naics_data_filepath, num_digits=None, starts_with=None):
    df = pd.read_csv(naics_data_filepath, usecols=[1,2], skiprows=1, dtype=str)
    df.columns = ['Code', 'Name']
    if starts_with is not None:
        df = df[df['Code'].astype(str).str.startswith(str(starts_with))]
    else:
        df = df
    if num_digits is not None:
        df = df[df['Code'].astype(str).str.len() == num_digits]
    else:
        df = df
    return df

def get_tracts(df_list):
    df_combined = pd.concat(df_list)
    ids = np.unique(df_combined[['poi_id', 'visitor_id']].values)
    return ids

def get_sparse_matrix(df, tracts): #row=home, col=poi
    num_tracts = len(tracts)
    directed_matrix = np.zeros((num_tracts, num_tracts),dtype=float)
    for index, row in df.iterrows():
        home_index = np.where(tracts == row['visitor_id'])
        poi_index = np.where(tracts == row['poi_id'])
        directed_matrix[home_index, poi_index] = row['count']
    sparse_matrix = csr_matrix(directed_matrix)
    return sparse_matrix

def build_matrices(df_list, tracts=None):
    mats = []
    if tracts is None:
        tracts = get_tracts(df_list)
    for df in df_list:
        mats.append(get_sparse_matrix(df, tracts))
    return mats

def get_dates(weekly_patterns_folder):
    date_list = []
    for file in os.listdir(weekly_patterns_folder):
        date = file[-14:-4]
        date_list.append(date)
    return sorted(date_list)

def get_total_movements(mat_list):
    movements = []
    for mat in mat_list:
        sums = np.sum(mat)
        movements.append(sums)
    movements = np.array(movements)
    return movements
    
def get_degrees_in(mat_list):
    degrees = []
    for mat in mat_list:
        sums = mat.sum(axis=0)
        sums = np.array(sums).flatten()
        degrees.append(sums)
    degrees = np.array(degrees)
    return degrees

def get_degrees_out(mat_list):
    degrees = []
    for mat in mat_list:
        sums = np.array(mat.sum(axis=1)).flatten()
        degrees.append(sums)
    degrees = np.array(degrees)
    return degrees

def plot_total_movements(total_movements, dates, axis):
    dates = [date[-5:] for date in dates]
    axis.plot(dates, total_movements, 'x', linestyle='', color='b')
    axis.set_xticks(dates)
    axis.set_xticklabels(dates, rotation='vertical')
    axis.set_xlabel(f'Week End-Date')
    axis.set_ylabel('Movements')

def plot_degrees(degrees, dates, axis):
    dates = [date[-5:] for date in dates]
    means = np.mean(degrees, axis=1)
    stds = np.std(degrees, axis=1)
    num_weeks = len(dates)
    num_tracts = np.shape(degrees)[1]
    axis.plot(dates, means, color='b', linewidth=1, label='Mean')
    axis.plot(dates, means+stds, alpha=0.2)
    axis.fill_between(dates, means, means+stds, color='b', alpha=0.2)
    axis.fill_between(dates, means, means-stds, color='b', alpha=0.2)
    for i in range(0,num_weeks):
        axis.scatter(
            np.full((num_tracts,),dates[i]),
            degrees[i,:],
            s=.3,
            color='c',
            label='CTs' if i==0 else None)
    axis.set_xticks(np.arange(0, len(dates)), dates, rotation='vertical', fontsize=10)
    axis.legend(fontsize=6, loc='upper right')
    axis.set_ylabel('Degrees')
    axis.set_xlabel('Week End-Date')

def generate_movement_data(weekly_patterns_folder, movement_data_folder, naics_code=''):
    dfs = create_dfs(weekly_patterns_folder)
    dfs = filter_dfs(dfs, 'visitor_id', '48201')
    dfs = filter_dfs(dfs, 'naics_code', naics_code)
    dfs = convert_to_tracts(dfs)
    tracts = get_tracts(dfs)
    mats = build_matrices(dfs, tracts)
    deg_in = get_degrees_in(mats)
    deg_out = get_degrees_out(mats)
    total_movements = get_total_movements(mats)
    dict = (
        {'Tract':tracts,
        'In-Degree':deg_in,
        'Out-Degree':deg_out,
        'Total Movements':total_movements}
        )
    if naics_code == '':
        filename = f'{movement_data_folder}/all.pkl'
    else:
        filename = f'{movement_data_folder}/{naics_code}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(dict, f)
    return None

def load_movement_data(movement_data_folder, naics_code=''):
    if naics_code == '0' or naics_code == '' or naics_code == 'all' or naics_code == '00':
        filename = f'{movement_data_folder}/all.pkl'
    else:
        filename = f'{movement_data_folder}/{naics_code}.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            net_stats = pickle.load(f)
    else:
        print('No movement data for NAICS: ', naics_code)
        return None
    return net_stats

def get_bins(array, num_bins):
    counts, bins = np.histogram(array, bins=num_bins, density=False)
    return bins

def calculate_zscores(total_movements, baseline_start, baseline_stop):
    baseline = total_movements[baseline_start:baseline_stop+1]
    baseline_mean = baseline.mean()
    baseline_std = baseline.std()
    entry = {
        'Baseline Mean': baseline_mean.round(2),
        'Baseline Std': baseline_std.round(2)
    }
    for i, movements in enumerate(total_movements):
        zscore = (movements - baseline_mean)/baseline_std
        entry[f'Z-score Week {i+1}'] = zscore.round(2)
    return pd.DataFrame([entry])

def get_distributions(total_movements, anomalous_week, baseline_start, baseline_stop, number_of_bins):
    baseline = total_movements[baseline_start:baseline_stop+1]
    anomaly = total_movements[anomalous_week]
    other = np.delete(
        total_movements,
        list(range(baseline_start, baseline_stop + 1)) + [anomalous_week]
        )
    bins = get_bins(np.delete(total_movements, anomalous_week), num_bins=number_of_bins)
    baseline_density, bins = np.histogram(baseline, bins, density=True)
    other_density, bins = np.histogram(other, bins, density=True)
    return bins, baseline_density, other_density 

def plot_distributions(bins, baseline_density, other_density, anomaly, axis):
    bin_centers = (bins[1]-bins[0])/2 + bins[:-1]
    width = (bins[1]-bins[0])
    axis.bar(
        bin_centers, 
        baseline_density, 
        width=width, 
        color='grey', 
        alpha=.1,
        label='Baseline'
        )
    axis.bar(
        bin_centers, 
        other_density, 
        width=width, 
        color='blue', 
        alpha=.1,
        label='Other'
        )
    axis.bar(
        anomaly,
        0.5 * max(np.max(baseline_density), np.max(other_density)),
        width=width,
        color='red',
        alpha=0.2,
        label='Anomaly'
        )
    axis.set_xlabel('Total Movements')
    axis.set_ylabel('Probability Density')
    axis.legend()
    return None

def plot_shading(anomalous_week, baseline_start, baseline_stop, axis):
    axis.axvspan(anomalous_week-.5, anomalous_week+.5, color='r', alpha=.2, linewidth=0)
    axis.axvspan(baseline_start-.5, baseline_stop+.5, color='grey', alpha=.2, linewidth=0)

def make_piechart(df, to_plot, wedgesize, startangle, radius, ax):
    df['Coarse Code'] = df['Coarse Code'].astype(str)
    df['Fine Code'] = df['Fine Code'].astype(str)
    #inner
    labels = [f"{code} {name[:10]} ..." for code, name in zip(df['Fine Code'], df['Fine Name'])]
    ax.pie(
        df[to_plot],
        startangle=startangle,
        wedgeprops={'width': wedgesize},
        labeldistance=1,         
        rotatelabels=True,
        radius = radius-wedgesize,
        colors = df['Color']
    )
    #outer
    coarse_df = df.groupby(['Coarse Code', 'Coarse Name', 'Coarse Color']).agg({
        'Total Movements': 'sum',
        'Number of POIs': 'sum'
    }).reset_index()
    labels = [f"{code} {name}" for code, name in zip(coarse_df['Coarse Code'], coarse_df['Coarse Name'])]
    ax.pie(
        coarse_df[to_plot],
        labels=labels,
        startangle=startangle,
        wedgeprops={'width': wedgesize},
        labeldistance=1.1,         
        rotatelabels=False,
        radius = radius,
        colors = coarse_df['Coarse Color'],
        textprops={'color': 'black'}
    )
    ax.set_title(f'{to_plot} By Category', x=.7, y=.8)
    return None

def get_sankey_data(
    source_df,
    target_df,
    aspect_to_plot,
    aspect_for_widths,
    aspect_for_label_values=None,
    criss_cross=True,
    sort_ascending=True):

    # Sort dataframes
    source_df = source_df.sort_values(by=aspect_to_plot, ascending=sort_ascending).reset_index(drop=True)
    source_map = {code: idx for idx, code in enumerate(source_df['Coarse Code'])} 
    if criss_cross == True:
        target_df = target_df.sort_values(by=aspect_to_plot).reset_index(drop=True)
    else:
        target_df['Source Aspect'] = target_df['Coarse Code'].map(source_map)
        target_df = target_df.sort_values(by=['Source Aspect', aspect_to_plot], ascending=[True, sort_ascending]).reset_index(drop=True)
        target_df.drop(columns=['Source Aspect'], inplace=True)

    # Nodes
    target_nodes =  list(range(len(source_df), len(source_df) + len(target_df)))
    source_nodes = list(target_df['Coarse Code'].map(source_map))

    # Node placement (used only in crisscross)
    x = [0.01]*len(source_df) + [.99]*len(target_df)
    y = list(np.linspace(0.04,1,len(source_df))) + list(np.linspace(0.1,1.03,len(target_df)))

    # Labels
    if aspect_for_label_values == None:
        source_labels = [f"{code} {name}" for code, name in zip(source_df['Coarse Code'], source_df['Coarse Name'])]
        target_labels = [f"{code} {name}" for code, name in zip(target_df['Fine Code'], target_df['Fine Name'])]
    else:
        source_labels = [f"{code} {name} ({val})" for code, name, val in zip(source_df['Coarse Code'], source_df['Coarse Name'], source_df[aspect_for_label_values])]
        target_labels = [f"{code} {name} ({val})" for code, name, val in zip(target_df['Fine Code'], target_df['Fine Name'], target_df[aspect_for_label_values])]
    labels = source_labels + target_labels

    # Colors and widths
    colors = target_df['Fine Color']
    node_widths = list(target_df[aspect_for_widths])

    if criss_cross == True:
        return source_nodes, target_nodes, node_widths, labels, colors, x, y
    else:
        return source_nodes, target_nodes, node_widths, labels, colors, None, None


def make_sankey(source_nodes, target_nodes, node_widths, labels, colors, x=None, y=None):
    fig = go.Figure(data=[go.Sankey(
    arrangement = 'freeform',
    node=dict(
        pad=15,  # Padding between nodes
        thickness=1,  # Thickness of the nodes
        line=dict(color='black', width=0.5),  # Outline of the nodes
        label=labels,  # Node labels
        x = x,
        y = y,
        color = 'black' 
    ),
    link=dict(
        source=source_nodes,  # Indices of source nodes
        target=target_nodes,  # Indices of target nodes
        value=node_widths,  # Values associated with the links (thickness)
        color = colors # Colors of the links
    ))])
    return fig
    
def plot_scatter_probability(distributions, bins, ax, color='b', alpha=1, label=''):
    centers = (bins[:-1] + bins[1:]) / 2
    ax.loglog(centers, distributions, color=color, alpha=alpha, marker='x', linestyle='', label=label)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Probability')
    return None

def get_degrees(week_start, week_stop, movements, anomalous_week=None):
    df = pd.DataFrame({
        'Tract': movements['Tract'],
        'Mean In-Degree': movements['In-Degree'][week_start:week_stop+1,:].mean(axis=0),
        'Mean Out-Degree': movements['Out-Degree'][week_start:week_stop+1,:].mean(axis=0)
    })
    if anomalous_week != None:
        df['Anomalous In-Degree'] = movements['In-Degree'][anomalous_week, :]
        df['Anomalous Out-Degree'] = movements['Out-Degree'][anomalous_week, :]
    return df
    
def merge_dfs(df1, df2, df3, omit_tracts):
    df1['Tract'] = df1['Tract'].astype(str)
    df2['Tract'] = df2['Tract'].astype(str)
    df3['Tract'] = df3['Tract'].astype(str)
    df = df1.merge(df2, on='Tract', how='outer').merge(df3, on='Tract', how='outer')
    df = df[~df['Tract'].isin(omit_tracts)]
    df = df.fillna(0)
    return df

def plot_correlations(x, y, ax, r=None, alpha=.2, color='b', label=''):
    ax.loglog(x, y, '.', markersize=4, alpha=.2, color=color, label=label)
    if r != None:
        ax.text(0.04, 0.9, f'r = {r:.6f}', transform=ax.transAxes, 
                    fontsize=10, verticalalignment='bottom', horizontalalignment='left', 
                    bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white'))
    return None


def run_regression(df, features, target, return_plot_data=False):
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2 = model.score(X_scaled, y)
    if return_plot_data:
        return y, y_pred, r2
    coefs = list(model.coef_) + [r2]
    names = features + ['R² (Model Fit)']
    return pd.DataFrame({'Name': names, 'Value': coefs})


def plot_observed_vs_predicted(df, features, target, ax, label='', color='b', alpha=0.3):
    y, y_pred, r2 = run_regression(df, features, target, return_plot_data=True)
    ax.loglog(y, y_pred, '.', linestyle='', color=color, alpha=alpha, markersize=3, label=label)
    ax.set_xlabel('Observed Value')
    ax.set_ylabel('Predicted Value')
    ax.text(0.05, 0.87, f'$R^2 = {r2:.4f}$', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white'))
    return None
