import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler
from scipy.stats import t as student_t

# read data
df = pd.read_excel('processed_housing_data.xlsx')
df = df.dropna()

lsoa_boundaries = gpd.read_file('LSOA_2011_EW_BFC_V3.shp')

if 'LSOA11CD' in lsoa_boundaries.columns:
    lsoa_boundaries = lsoa_boundaries.rename(columns={'LSOA11CD': 'lsoa11cd'})
lsoa_boundaries['lsoa11cd'] = lsoa_boundaries['lsoa11cd'].astype(str).str.strip()
df['lsoa11cd'] = df['lsoa11cd'].astype(str).str.strip()

london_codes = pd.read_excel('LSOA_London.xlsx')
if 'LSOA11CD' not in london_codes.columns:
    london_codes = london_codes.rename(columns={london_codes.columns[0]: 'LSOA11CD'})
london_codes = london_codes[['LSOA11CD']].rename(columns={'LSOA11CD': 'lsoa11cd'})
london_codes['lsoa11cd'] = london_codes['lsoa11cd'].astype(str).str.strip()

gdf = lsoa_boundaries[lsoa_boundaries['lsoa11cd'].isin(df['lsoa11cd'])].copy()
gdf = gdf.merge(df, on='lsoa11cd', how='inner')

# use the coordinate system: British Grid
if gdf.crs is None:
    gdf.set_crs('EPSG:27700', inplace=True)
elif gdf.crs.to_string() != 'EPSG:27700':
    gdf = gdf.to_crs('EPSG:27700')


# data preparation
y = gdf['good_prop'].values.reshape(-1, 1)
feature_columns = ['white_prop', 'owned_prop', 'independent_prop', 'semi_connected_prop', 'IMD_decile']
X = gdf[feature_columns].values

gdf_centroids = gdf.geometry.centroid
coords = np.column_stack([gdf_centroids.x, gdf_centroids.y])

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# bandwidth
bw_selector = Sel_BW(coords, y_scaled, X_scaled, kernel='gaussian')
optimal_bw = bw_selector.search(criterion='AICc')



# model fits
gwr_model = GWR(coords, y_scaled, X_scaled, bw=optimal_bw, kernel='gaussian')
gwr_results = gwr_model.fit()




# model performance results
y_mean = np.mean(y_scaled)
ss_tot = np.sum((y_scaled - y_mean) ** 2)
ss_res = np.sum(gwr_results.resid_response ** 2)
r_squared = 1 - (ss_res / ss_tot)
adj_r_squared = 1 - (1 - r_squared) * (gwr_results.n - 1) / (gwr_results.n - gwr_results.tr_S)

MAE = float(np.mean(np.abs(scaler_y.inverse_transform(gwr_results.resid_response.reshape(-1,1)))))

fitted_values = scaler_y.inverse_transform(gwr_results.predy.reshape(-1, 1)).flatten()
residuals = gdf['good_prop'].values - fitted_values
MAE = float(np.mean(np.abs(residuals)))
RMSE = float(np.sqrt(np.mean(residuals**2)))

# local coefficients
local_coefficients = gwr_results.params 
local_tvalues = gwr_results.tvalues
df_dof = max(1, int(round(gwr_results.n - gwr_results.tr_S)))
local_pvalues = 2 * (1 - student_t.cdf(np.abs(local_tvalues), df=df_dof))

coef_columns = ['intercept'] + feature_columns
for i, col in enumerate(coef_columns):
    gdf[f'{col}_coef'] = local_coefficients[:, i]
    gdf[f'{col}_tval'] = local_tvalues[:, i]
    gdf[f'{col}_pval'] = local_pvalues[:, i]




sigma_y = float(scaler_y.scale_[0]); mu_y = float(scaler_y.mean_[0])
sigma_x = scaler_X.scale_.astype(float); mu_x = scaler_X.mean_.astype(float)

b0_std = local_coefficients[:, 0]
b_vars_std = local_coefficients[:, 1:] 
b_vars_orig = b_vars_std * (sigma_y / sigma_x) 
intercept_orig = mu_y + sigma_y * b0_std - np.sum(b_vars_orig * mu_x, axis=1)

for j, col in enumerate(feature_columns):
    gdf[f'{col}_coef_orig'] = b_vars_orig[:, j]
gdf['intercept_coef_orig'] = intercept_orig

# residuals
gdf['fitted_values'] = fitted_values
gdf['residuals'] = residuals
gdf['std_residuals'] = gdf['residuals'] / np.std(gdf['residuals'])





# statistics summary
coef_stats = pd.DataFrame()
for col in coef_columns:
    cc = f'{col}_coef'
    coef_stats[col] = [
        gdf[cc].mean(), gdf[cc].std(), gdf[cc].min(),
        gdf[cc].quantile(0.25), gdf[cc].median(),
        gdf[cc].quantile(0.75), gdf[cc].max()
    ]
coef_stats.index = ['Mean', 'Std', 'Min', 'Q1', 'Median', 'Q3', 'Max']

lsoa_london_background = lsoa_boundaries[lsoa_boundaries['lsoa11cd'].isin(london_codes['lsoa11cd'])].copy()
if lsoa_london_background.crs is None:
    lsoa_london_background.set_crs('EPSG:27700', inplace=True)
if lsoa_london_background.crs.to_string() != 'EPSG:4326':
    lsoa_london_background = lsoa_london_background.to_crs('EPSG:4326')

gdf_plot = gdf.to_crs('EPSG:4326')

plot_columns = [
    'white_prop_coef', 'owned_prop_coef', 'independent_prop_coef',
    'semi_connected_prop_coef', 'IMD_decile_coef', 'std_residuals'
]
plot_titles = [
    'White Population Ratio Coefficient', 'Owned Housing Ratio Coefficient',
    'Detached Housing Coefficient', 'Semi-detached Housing Coefficient',
    'IMD Decile Coefficient', 'Standardized Residuals'
]

# create maps
for col, title in zip(plot_columns, plot_titles):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    lsoa_london_background.plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.1)
    gdf_plot.plot(column=col, cmap='RdBu_r', ax=ax,
                  edgecolor='white', linewidth=0.1, alpha=0.8,
                  legend=True, legend_kwds={'shrink': 0.6})
    ax.set_axis_off()
    plt.tight_layout()

# histograms
for col, title in zip(plot_columns[:-1], plot_titles[:-1]):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    gdf[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
    ax.axvline(gdf[col].mean(), color='red', linestyle='--', label=f'Mean: {gdf[col].mean():.3f}')
    ax.set_xlabel('Coefficient Value'); ax.set_ylabel('Frequency')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()




# significant statistics summary
significance_rows = []
for col in feature_columns:
    significant_count = int(np.sum(np.abs(gdf[f'{col}_tval']) > 1.96))
    total_count = int(len(gdf))
    pct = significant_count / total_count * 100
    significance_rows.append({
        'Variable': col,
        'Significant_areas': significant_count,
        'Total_areas': total_count,
        'Percentage_significant': pct
    })
significance_df = pd.DataFrame(significance_rows)


# identify the 5 top and bottom significant areas for each LSOA
sig_detail_rows = []
for col in feature_columns:
    coef_col = f'{col}_coef'; tval_col = f'{col}_tval'
    significant_areas = gdf[np.abs(gdf[tval_col]) > 1.96].copy()
    if len(significant_areas) > 0:
        top_areas = significant_areas.nlargest(5, coef_col)
        for _, row in top_areas.iterrows():
            sig_detail_rows.append({'Variable': col, 'Type': 'Highest',
                                    'LSOA_Code': row['lsoa11cd'],
                                    'Coefficient': row[coef_col], 'T_value': row[tval_col]})
        bottom_areas = significant_areas.nsmallest(5, coef_col)
        for _, row in bottom_areas.iterrows():
            sig_detail_rows.append({'Variable': col, 'Type': 'Lowest',
                                    'LSOA_Code': row['lsoa11cd'],
                                    'Coefficient': row[coef_col], 'T_value': row[tval_col]})
significant_areas_df = pd.DataFrame(sig_detail_rows)



model_summary = {
    'Number_of_observations': int(gwr_results.n),
    'Effective_degrees_of_freedom': float(gwr_results.tr_S),
    'Residual_degrees_of_freedom': float(gwr_results.tr_STS),
    'AIC': float(gwr_results.aic),
    'AICc': float(gwr_results.aicc),
    'R_squared': float(r_squared),
    'Adjusted_R_squared': float(adj_r_squared),
    'MAE': float(MAE),
    'RMSE': float(RMSE),
}
model_summary_df = pd.DataFrame(list(model_summary.items()), columns=['Metric', 'Value'])



