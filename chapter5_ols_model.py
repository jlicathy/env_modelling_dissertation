import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# read data
df = pd.read_excel('processed_housing_data.xlsx')

lsoa_boundaries = gpd.read_file('LSOA_2011_EW_BFC_V3.shp')

if 'LSOA11CD' in lsoa_boundaries.columns:
    lsoa_boundaries = lsoa_boundaries.rename(columns={'LSOA11CD': 'lsoa11cd'})
lsoa_boundaries['lsoa11cd'] = lsoa_boundaries['lsoa11cd'].astype(str).str.strip()
df['lsoa11cd'] = df['lsoa11cd'].astype(str).str.strip()

gdf = lsoa_boundaries[lsoa_boundaries['lsoa11cd'].isin(df['lsoa11cd'])].copy()
gdf = gdf.merge(df, on='lsoa11cd', how='inner')



# modelling process


y = gdf['good_prop'].values

feature_columns = [
    'white_prop',
    'owned_prop',
    'independent_prop',
    'semi_connected_prop',
    'IMD_decile'
]

X = gdf[feature_columns].values



scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()



X_scaled_sm = sm.add_constant(X_scaled)
ols_model = sm.OLS(y_scaled, X_scaled_sm).fit()

# results summary
print(f"Number of observations: {len(y_scaled)}")
print(f"R²: {ols_model.rsquared:.4f}")
print(f"Adjusted R²: {ols_model.rsquared_adj:.4f}")
print(f"AIC: {ols_model.aic:.2f}")
print(f"F-statistic: {ols_model.fvalue:.2f} (p-value: {ols_model.f_pvalue:.6f})")


def get_significance(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""

# create result table
results_table = pd.DataFrame({
    'Variable': ['Constant'] + feature_columns,
    'Coefficient': ols_model.params,
    'P-value': ols_model.pvalues,
    'Significance': [get_significance(p) for p in ols_model.pvalues]
})

print(results_table.to_string(index=False, float_format='%.4f'))

