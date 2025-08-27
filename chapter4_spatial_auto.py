import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.patches as mpatches

from libpysal.weights import Queen, Rook
from esda import Moran, Moran_Local
from splot.esda import moran_scatterplot, lisa_cluster, plot_moran

# MC process def
np.random.seed(999)


plt.style.use('default')
sns.set_palette("viridis")

# read data
df = pd.read_excel('processed_housing_data.xlsx')

london_codes = pd.read_excel('LSOA_London.xlsx')

london_codes = london_codes[['LSOA11CD']].rename(columns={'LSOA11CD': 'lsoa11cd'})
london_codes['lsoa11cd'] = london_codes['lsoa11cd'].astype(str).str.strip()

lsoa_boundaries = gpd.read_file('LSOA_2011_EW_BFC_V3.shp')

if 'LSOA11CD' in lsoa_boundaries.columns:
    lsoa_boundaries = lsoa_boundaries.rename(columns={'LSOA11CD': 'lsoa11cd'})
lsoa_boundaries['lsoa11cd'] = lsoa_boundaries['lsoa11cd'].astype(str).str.strip()

lsoa_gla_background = lsoa_boundaries[lsoa_boundaries['lsoa11cd'].isin(london_codes['lsoa11cd'])].copy()

df['lsoa11cd'] = df['lsoa11cd'].astype(str).str.strip()
gdf_analysis = lsoa_boundaries[lsoa_boundaries['lsoa11cd'].isin(df['lsoa11cd'])].copy()
gdf_analysis = gdf_analysis.merge(df, on='lsoa11cd', how='inner')

# WGS84 setting
if gdf_analysis.crs is None:
    gdf_analysis.set_crs('EPSG:27700', inplace=True) 
if gdf_analysis.crs != 'EPSG:4326':
    gdf_analysis = gdf_analysis.to_crs('EPSG:4326')

if lsoa_gla_background.crs != 'EPSG:4326':
    lsoa_gla_background = lsoa_gla_background.to_crs('EPSG:4326')


# Queen contiguity weights matrix

w_queen = Queen.from_dataframe(gdf_analysis, silence_warnings=True)
print(f"Average neighbors per area: {w_queen.mean_neighbors:.2f}")
    
    # check islands
islands = w_queen.islands
if len(islands) > 0:
   print(f"{len(islands)} isolated areas detected")
   gdf_analysis = gdf_analysis.drop(gdf_analysis.index[islands])
   w_queen = Queen.from_dataframe(gdf_analysis, silence_warnings=True)
   print(f"After removing islands: {w_queen.n} areas, {w_queen.s0} links")
    
    # standardlized
w_queen.transform = 'r'
    




# Global Moran's I

analysis_vars = {
    'good_prop': 'Good House Proportion',
    'bad_prop': 'Poor House Proportion', 
    'IMD_decile': 'IMD Deprivation Decile',
    'independent_prop': 'Independent House Proportion',
    'semi_connected_prop': 'Semi-connected House Proportion',
    'full_connected_prop': 'Full-connected House Proportion',
    'owned_prop': 'Owner-occupied Proportion',
    'white_prop': 'White Population Proportion'
}

global_results = []

for var, var_name in analysis_vars.items():
    # Global Moran's I
    y = gdf_analysis[var].values
    moran = Moran(y, w_queen)
    
    # significance level
    if moran.p_norm < 0.001:
        significance = "***"
        sig_text = "Highly Significant"
    elif moran.p_norm < 0.01:
        significance = "**"
        sig_text = "Significant"
    elif moran.p_norm < 0.05:
        significance = "*"
        sig_text = "Significant"
    else:
        significance = "ns"
        sig_text = "Not Significant"
    
    # spatial distribution 
    if moran.I > moran.EI:
        pattern = "Positive Clustering"
    else:
        pattern = "Spatial Dispersion"
    
    global_results.append({
        'Variable': var_name,
        'Morans_I': round(moran.I, 4),
        'Expected_I': round(moran.EI, 4),
        'Z_score': round(moran.z_norm, 4),
        'P_value': f"{moran.p_norm:.2e}",
        'Significance': significance,
        'Interpretation': f"{pattern} ({sig_text})"
    })

# create a result summary
moran_results_df = pd.DataFrame(global_results)

print(moran_results_df.to_string(index=False))






# 3. LISA

lisa_vars = ['good_prop', 'bad_prop', 'independent_prop', 'owned_prop', 'white_prop']

for var in lisa_vars:
    var_name = analysis_vars[var]
    print(f"\nLISA Analysis for {var_name}:")
    
    # calculate
    y = gdf_analysis[var].values
    lisa = Moran_Local(y, w_queen)
    
    # add results to dataframe
    gdf_analysis[f'{var}_lisa_i'] = lisa.Is
    gdf_analysis[f'{var}_lisa_p'] = lisa.p_sim
    gdf_analysis[f'{var}_lisa_q'] = lisa.q
    
    # use the 90% CI
    gdf_analysis[f'{var}_lisa_sig'] = (lisa.p_sim < 0.10)
    
    # LISA cluster type
    lisa_labels = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL', 0: 'Not Significant'}
    gdf_analysis[f'{var}_lisa_cluster'] = gdf_analysis[f'{var}_lisa_q'].map(lisa_labels)
    gdf_analysis.loc[~gdf_analysis[f'{var}_lisa_sig'], f'{var}_lisa_cluster'] = 'Not Significant'
    
    # cluster count
    cluster_counts = gdf_analysis[f'{var}_lisa_cluster'].value_counts()
    print(f"  LISA Cluster Distribution:")
    for cluster, count in cluster_counts.items():
        print(f"    {cluster}: {count} areas")
    
    # significant cluster count
    sig_count = gdf_analysis[f'{var}_lisa_sig'].sum()
    print(f"  Significant clusters (p < 0.10): {sig_count}/{len(gdf_analysis)} ({sig_count/len(gdf_analysis)*100:.1f}%)")



# LISA map

lisa_colors = {
    'HH': '#d7191c', 
    'HL': '#abd9e9', 
    'LH': '#fdae61', 
    'LL': '#2c7bb6', 
    'Not Significant': '#f7f7f7'
}


for var in lisa_vars:
    var_name = analysis_vars[var]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    lsoa_gla_background.plot(ax=ax, color="#f0f0f0", edgecolor='white', linewidth=0.1)
    
    cluster_col = f'{var}_lisa_cluster'
    for cluster_type in lisa_colors.keys():
        if cluster_type in gdf_analysis[cluster_col].values:
            subset = gdf_analysis[gdf_analysis[cluster_col] == cluster_type]
            if not subset.empty:
                subset.plot(ax=ax, color=lisa_colors[cluster_type], 
                           edgecolor='white', linewidth=0.1, alpha=0.8)
    
    legend_handles = []
    for cluster_type, color in lisa_colors.items():
        if cluster_type in gdf_analysis[cluster_col].values:
            count = (gdf_analysis[cluster_col] == cluster_type).sum()
            label = f'{cluster_type} (n={count})'
            legend_handles.append(mpatches.Patch(color=color, label=label))
    
    ax.legend(handles=legend_handles, title='LISA Clusters',
              loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
              fontsize=10, title_fontsize=11, frameon=True)
    
    ax.axis('off')
    



lisa_summary = []
for var in lisa_vars:
    cluster_counts = gdf_analysis[f'{var}_lisa_cluster'].value_counts()
    for cluster_type, count in cluster_counts.items():
        lisa_summary.append({
            'Variable': analysis_vars[var],
            'Cluster_Type': cluster_type,
            'Count': count,
            'Percentage': round(count / len(gdf_analysis) * 100, 1)
        })

lisa_summary_df = pd.DataFrame(lisa_summary)


print("\nLISA Cluster Summary:")
for var in lisa_vars:
    var_name = analysis_vars[var]
    print(f"\n{var_name}:")
    var_data = lisa_summary_df[lisa_summary_df['Variable'] == var_name]
    for _, row in var_data.iterrows():
        print(f"  {row['Cluster_Type']}: {row['Count']} areas ({row['Percentage']}%)")


