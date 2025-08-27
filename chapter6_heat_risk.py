import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


# read data
UA_GPKG    = "UK001L3_LONDON_UA2018_v013.gpkg" 
LSOA_SHP   = "LSOA_2011_EW_BFC_V3.shp"
ATTR_XLSX  = "canopy_lsoa.xlsx"
LONDON_XLS = "LSOA_London.xlsx"


def pick(cols, cands):
    for c in cands:
        if c in cols:
            return c
    raise ValueError(f"cannot find {cands} in {list(cols)}")

# read UA layers
ua = gpd.read_file(UA_GPKG, layer="UK001L3_LONDON_UA2018").to_crs(27700)
# not sure the code name
UA_CODE = pick(ua.columns, ["code_2018","CODE_2018","Item2018","ITEM2018"])




# London lSOA shape
lsoa = gpd.read_file(LSOA_SHP).to_crs(27700)
LSOA_CODE = pick(lsoa.columns, ["lsoa11cd","LSOA11CD","lsoa21cd","LSOA21CD","LSOA_CODE","lsoa_code"])

london_df = pd.read_excel(LONDON_XLS)
LONDON_CODE = pick(london_df.columns, ["LSOA11CD","lsoa11cd","LSOA21CD","lsoa21cd","LSOA_CODE","lsoa_code"])
london_codes = london_df[LONDON_CODE].astype(str).str.strip().tolist()

lsoa[LSOA_CODE] = lsoa[LSOA_CODE].astype(str).str.strip()
lsoa = lsoa[lsoa[LSOA_CODE].isin(london_codes)].copy()



# read canopy data
attrs = pd.read_excel(ATTR_XLSX)
ATTR_CODE = pick(attrs.columns, ["lsoa11cd","LSOA11CD","lsoa21cd","LSOA21CD","LSOA_CODE","lsoa_code"])
CANOPYCOL = pick(attrs.columns, ["canopy_per","canopy_%","canopy"])

# choose the impervious layers of UA
impervious_codes = ["11100","11210","11220","11230","11240","12100","12210","12220"]
ua_imp = ua[ua[UA_CODE].astype(str).isin(impervious_codes)].copy()



# 5) aggregate and calculations
intersect = gpd.overlay(
    lsoa[[LSOA_CODE, "geometry"]],
    ua_imp[[UA_CODE, "geometry"]],
    how="intersection"
)
intersect["__inter_area"] = intersect.geometry.area
imp_area = intersect.groupby(LSOA_CODE, as_index=False)["__inter_area"].sum().rename(columns={"__inter_area":"imperv_area"})

lsoa = lsoa.merge(imp_area, on=LSOA_CODE, how="left")
lsoa["imperv_area"] = lsoa["imperv_area"].fillna(0.0)
lsoa["impervious_per"] = (lsoa["imperv_area"] / lsoa["__area_total"]) * 100

g = lsoa.merge(
    attrs[[ATTR_CODE, CANOPYCOL]].rename(columns={ATTR_CODE: LSOA_CODE, CANOPYCOL: "canopy_per"}),
    on=LSOA_CODE, how="left"
)

# avoid NAN
g["impervious_per"] = g["impervious_per"].fillna(0.0)
g["canopy_per"]     = g["canopy_per"].fillna(g["canopy_per"].median())



# calculate heat hazard index
def z(s):
    s = s.astype(float)
    return (s - s.mean())/s.std(ddof=0)

g["HazardIndex"] = (-z(g["canopy_per"]) + z(g["impervious_per"])) / 2.0

g["Hazard_q"] = pd.qcut(g["HazardIndex"], 5, labels=[1,2,3,4,5], duplicates="drop").astype("Int64")



fig, ax = plt.subplots(figsize=(10, 8))
lsoa.boundary.plot(ax=ax, color="lightgrey", linewidth=0.1)
g.plot(
    column="Hazard_q", cmap="YlOrRd",
    linewidth=0.05, edgecolor="white",
    legend=True, ax=ax, categorical=True
)
# define colorbar
cbar_ax = ax.get_figure().axes[-1]
cbar_ax.set_ylabel("Quintiles", fontsize=10)

ax.set_axis_off()
plt.show()





# heat risk analysis

# read the social vulunearble data
SOCIO_XLSX = "processed_housing_data.xlsx"
socio = pd.read_excel(SOCIO_XLSX)
socio["lsoa11cd"] = socio["lsoa11cd"].astype(str).str.strip()

# z-score
def z(s):
    s = s.astype(float)
    return (s - s.mean())/s.std(ddof=0)

# 11-IMD: because here I use the IMD Decile (1-10)
socio["V_IMD"]      = 11 - socio["IMD_decile"].astype(float)
socio["V_rented"]   = socio["rented_prop"].astype(float)
socio["V_minority"] = socio["minority_prop"].astype(float)

socio["VulIndex"] = np.nanmean(
    [z(socio["V_IMD"]), z(socio["V_rented"]), z(socio["V_minority"])],
    axis=0
)

# aggregate with g
g2 = g.merge(socio[["lsoa11cd","VulIndex"]],
             left_on=LSOA_CODE, right_on="lsoa11cd", how="left")

# still avoid NAN
g2["VulIndex"] = g2["VulIndex"].fillna(g2["VulIndex"].median())


# Heat Risk calculation and mapping
g2["RiskIndex"] = z(g2["HazardIndex"]) + z(g2["VulIndex"])
g2["Risk_q"]    = pd.qcut(g2["RiskIndex"], 5, labels=[1,2,3,4,5], duplicates="drop").astype("Int64")

fig, ax = plt.subplots(figsize=(10, 8))
g.boundary.plot(ax=ax, color="lightgrey", linewidth=0.1)
g2.plot(
    column="Risk_q", cmap="YlOrRd",
    linewidth=0.05, edgecolor="white",
    legend=True, ax=ax, categorical=True
)
cbar_ax = ax.get_figure().axes[-1]
cbar_ax.set_ylabel("Quintiles", fontsize=10)
ax.set_axis_off()
plt.show()