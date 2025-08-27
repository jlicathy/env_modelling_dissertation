import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches


# Set plot style
plt.style.use('default')
sns.set_palette("viridis")


# read data
df = pd.read_excel('processed_housing_data.xlsx')

london_codes = pd.read_excel('LSOA_London.xlsx')
if 'LSOA11CD' not in london_codes.columns:
    raise ValueError("not found LSOA11CD")
london_codes = london_codes[['LSOA11CD']].rename(columns={'LSOA11CD': 'lsoa11cd'})
london_codes['lsoa11cd'] = london_codes['lsoa11cd'].astype(str).str.strip()

lsoa_boundaries = gpd.read_file('LSOA_2011_EW_BFC_V3.shp')

if 'LSOA11CD' in lsoa_boundaries.columns:
    lsoa_boundaries = lsoa_boundaries.rename(columns={'LSOA11CD': 'lsoa11cd'})
lsoa_boundaries['lsoa11cd'] = lsoa_boundaries['lsoa11cd'].astype(str).str.strip()


# select Greater London boundary and use left join
lsoa_gla = lsoa_boundaries[lsoa_boundaries['lsoa11cd'].isin(london_codes['lsoa11cd'])].copy()

gdf = lsoa_gla.merge(df, on='lsoa11cd', how='left')

# WGS84 coordinates
if gdf.crs is None:
    gdf.set_crs(lsoa_gla.crs, inplace=True)
if gdf.crs != 'EPSG:4326':
    gdf = gdf.to_crs('EPSG:4326')






# spatial distribution map

# define the missing style: grey represents no pre-1900 houses in this LSOA
missing_style = {'color': '#D9D9D9', 'edgecolor': 'white', 'hatch': None, 'label': 'No pre-1900 homes'}

def draw_choropleth(column, cmap, filename, legend_label=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    gdf.plot(column=column, cmap=cmap, linewidth=0.1, ax=ax,
             edgecolor='white', legend=True,
             legend_kwds={'shrink': 0.8, 'label': legend_label} if legend_label else {'shrink': 0.8},
             missing_kwds=missing_style)
    ax.axis('off')

# Good house proportion
draw_choropleth('good_prop', 'Greens', 'map_good_prop.png', legend_label='Good House Proportion')

# Bad house proportion
draw_choropleth('bad_prop', 'Reds', 'map_bad_prop.png', legend_label='Poor House Proportion')

# IMD decile
draw_choropleth('IMD_decile', 'plasma', 'map_imd_decile.png', legend_label='IMD Decile (1=Most Deprived, 10=Least)')

# Dominant Building Type
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
lsoa_gla.to_crs(gdf.crs).plot(ax=ax, color='#D9D9D9', edgecolor='white', linewidth=0.1)

def get_dominant_building_type(row):
    if pd.isna(row.get('independent_prop')) or pd.isna(row.get('semi_connected_prop')) or pd.isna(row.get('full_connected_prop')):
        return np.nan
    types = {
        'independent': row['independent_prop'],
        'semi_connected': row['semi_connected_prop'],
        'full_connected': row['full_connected_prop']
    }
    return max(types, key=types.get)

gdf['dominant_building_type'] = gdf.apply(get_dominant_building_type, axis=1)

building_type_colors = {
    'independent': '#E41A1C',    
    'semi_connected': '#377EB8', 
    'full_connected': '#4DAF4A' 
}
for t, color in building_type_colors.items():
    sub = gdf[gdf['dominant_building_type'] == t]
    if not sub.empty:
        sub.plot(ax=ax, color=color, edgecolor='white', linewidth=0.1, alpha=0.95)

legend_handles = [
    mpatches.Patch(color=building_type_colors['independent'], label='independent'),
    mpatches.Patch(color=building_type_colors['semi_connected'], label='semi-connected'),
    mpatches.Patch(color=building_type_colors['full_connected'], label='full-connected')
]
ax.legend(handles=legend_handles, title="Dominant Building Type",
          loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
          fontsize=10, title_fontsize=11, frameon=True)
ax.axis('off')






# spatial cluster

# LSOA center coordinates
gdf['centroid'] = gdf.geometry.centroid
gdf['centroid_x'] = gdf.centroid.x
gdf['centroid_y'] = gdf.centroid.y

cluster_vars = ['good_prop', 'bad_prop', 'IMD_decile',
                'independent_prop', 'semi_connected_prop', 'full_connected_prop',
                'owned_prop', 'white_prop', 'centroid_x', 'centroid_y']

gdf_clu = gdf.dropna(subset=cluster_vars).copy()

# standardlized
scaler = StandardScaler()
cluster_data = scaler.fit_transform(gdf_clu[cluster_vars])

# K-means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
gdf_clu['cluster'] = kmeans.fit_predict(cluster_data)

# merge back results
gdf = gdf.merge(gdf_clu[['lsoa11cd','cluster']], on='lsoa11cd', how='left')

cluster_ids = sorted(gdf_clu['cluster'].unique())
palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00'] 
cluster_color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}

# cluster summary print info
cluster_summary = gdf_clu.groupby('cluster').agg({
    'good_prop': ['mean', 'std', 'count'],
    'bad_prop': ['mean', 'std'],
    'IMD_decile': ['mean', 'std'],
    'independent_prop': ['mean', 'std'],
    'semi_connected_prop': ['mean', 'std'],
    'full_connected_prop': ['mean', 'std'],
    'owned_prop': ['mean', 'std'],
    'white_prop': ['mean', 'std']
}).round(3)
print("Cluster Summary Statistics:")
print(cluster_summary)

# cluster map
fig, ax = plt.subplots(1, 1, figsize=(9, 7))
lsoa_gla.to_crs(gdf.crs).plot(ax=ax, color='#D9D9D9', edgecolor='white', linewidth=0.1)
for cid in cluster_ids:
    sub = gdf_clu[gdf_clu['cluster'] == cid]
    if not sub.empty:
        sub.plot(ax=ax, color=cluster_color_map[cid], edgecolor='white', linewidth=0.1, alpha=0.95)
# legend setting
handles = [mpatches.Patch(color=cluster_color_map[cid], label=f'Cluster {cid}') for cid in cluster_ids]
ax.legend(handles=handles, title='Cluster',
          loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
          fontsize=10, title_fontsize=11, frameon=True)
ax.axis('off')

# cluster bar chart
cluster_means = gdf_clu.groupby('cluster')['good_prop'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
bar_colors = [cluster_color_map[cid] for cid in cluster_means.index]
ax.bar(range(len(cluster_means)), cluster_means.values, color=bar_colors)
ax.set_xlabel('Cluster')
ax.set_ylabel('Good House Proportion')
ax.set_xticks(range(len(cluster_means)))
ax.set_xticklabels([f'Cluster {i}' for i in cluster_means.index])
