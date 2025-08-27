import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# read data
def load_processed_data(file_path):
    df = pd.read_excel(file_path)
    return df

# basic desprictive statistics
def basic_descriptive_stats(df):
    
    # 1. count
    count_vars = ['good_count', 'bad_count', 'total_count',
                  'independent_count', 'semi_connected_count', 'full_connected_count',
                  'white_count', 'minority_count', 'owned_count', 'rented_count']
    
    count_stats = df[count_vars].describe()
    
    # 2. proportion
    prop_vars = ['good_prop', 'bad_prop',
                 'independent_prop', 'semi_connected_prop', 'full_connected_prop',
                 'white_prop', 'minority_prop', 'owned_prop', 'rented_prop']
    
    prop_stats = df[prop_vars].describe()
    
    # 3. IMD decile
    imd_vars = ['IMD_decile']
    imd_stats = df[imd_vars].describe()
    
    # 4. skewness and kurtosis
    all_numeric_vars = count_vars + prop_vars + imd_vars
    skew_kurt = pd.DataFrame({
        'Skewness': df[all_numeric_vars].skew(),
        'Kurtosis': df[all_numeric_vars].kurtosis()
    })
    
    return count_stats, prop_stats, imd_stats, skew_kurt

# Pearson correlation analysis
def pearson_correlation_analysis(df):
    
    explanatory_vars = ['independent_prop', 'semi_connected_prop', 'full_connected_prop',
                       'white_prop', 'minority_prop', 'owned_prop', 'rented_prop',
                       'IMD_decile']
    
    dependent_var = 'good_prop'
    
    # correlation matrix
    all_vars = [dependent_var] + explanatory_vars
    correlation_matrix = df[all_vars].corr()
    
    # create variable mapping to change the names
    var_name_mapping = {
        'good_prop': 'Good House Proportion',
        'independent_prop': 'Independent House Proportion',
        'semi_connected_prop': 'Semi-connected House Proportion', 
        'full_connected_prop': 'Full-connected House Proportion',
        'white_prop': 'White Proportion',
        'minority_prop': 'Minority Proportion',
        'owned_prop': 'Owned Household Proportion',
        'rented_prop': 'Rented Household Proportion',
        'IMD_decile': 'IMD Deprivation Deciles'
    }
    
    correlation_matrix.index = [var_name_mapping.get(var, var) for var in correlation_matrix.index]
    correlation_matrix.columns = [var_name_mapping.get(var, var) for var in correlation_matrix.columns]
    
    
    # correlation analysis
    dependent_correlations = df[explanatory_vars + [dependent_var]].corr()[dependent_var].drop(dependent_var)
    dependent_correlations_sorted = dependent_correlations.abs().sort_values(ascending=False)
    
    correlation_summary = pd.DataFrame({
        'Variable': dependent_correlations_sorted.index,
        'Correlation_Coefficient': dependent_correlations[dependent_correlations_sorted.index].values,
        'Absolute_Value': dependent_correlations_sorted.values,
        'Strength': ['Very Strong (>0.8)' if abs(x) > 0.8 
                    else 'Strong (0.6-0.8)' if abs(x) > 0.6 
                    else 'Moderate (0.4-0.6)' if abs(x) > 0.4 
                    else 'Weak (0.2-0.4)' if abs(x) > 0.2 
                    else 'Very Weak (<0.2)' for x in dependent_correlations[dependent_correlations_sorted.index]]
    })
    
    
    # significant test
    from scipy.stats import pearsonr

    significance_results = []
    for var in explanatory_vars:
        corr_coef, p_value = pearsonr(df[var].dropna(), df[dependent_var].dropna())
        display_name = var_name_mapping.get(var, var)
        significance_results.append({
            'Variable': display_name,
            'Correlation': corr_coef,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No',
            'Significance_Level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
    
    significance_df = pd.DataFrame(significance_results)
    significance_df = significance_df.reindex(significance_df['Correlation'].abs().sort_values(ascending=False).index)

    
    # 4. matrix heat map
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                         square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt='.3f')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    
    # 5. correlation bar chart
    plt.figure(figsize=(16, 8))
    colors = ['red' if x < 0 else 'blue' for x in dependent_correlations[dependent_correlations_sorted.index]]
    bars = plt.barh(range(len(dependent_correlations_sorted)), 
                   dependent_correlations[dependent_correlations_sorted.index], 
                   color=colors, alpha=0.7)
    
    y_labels = [var_name_mapping.get(var, var) for var in dependent_correlations_sorted.index]
    plt.yticks(range(len(dependent_correlations_sorted)), y_labels)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    plt.xlim(-0.20, 0.25)
    
    for i, (bar, val) in enumerate(zip(bars, dependent_correlations[dependent_correlations_sorted.index])):
        plt.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                va='center', ha='left' if val >= 0 else 'right', fontsize=10)
    
    plt.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.08)
      
    return {
        'full_correlation_matrix': correlation_matrix,
        'dependent_correlations': correlation_summary,
        'significance_results': significance_df
    }


# building type reclassification
def housing_type_analysis(df):
    
    housing_summary = pd.DataFrame({
        'independent': [df['independent_count'].sum(), df['independent_prop'].mean(), df['independent_prop'].std()],
        'semi-connected': [df['semi_connected_count'].sum(), df['semi_connected_prop'].mean(), df['semi_connected_prop'].std()],
        'full-connected': [df['full_connected_count'].sum(), df['full_connected_prop'].mean(), df['full_connected_prop'].std()]
    }, index=['Total Count', 'Mean Proportion', 'Std Proportion'])
    
    plt.figure(figsize=(10, 6))
    housing_counts = [df['independent_count'].sum(), 
                     df['semi_connected_count'].sum(), 
                     df['full_connected_count'].sum()]
    
    bars = plt.bar(['independent', 'semi-connected', 'full-connected'], housing_counts, 
                   color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Housing Count')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, housing_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(housing_counts)*0.01, 
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plt.figure(figsize=(10, 6))
    housing_props = df[['independent_prop', 'semi_connected_prop', 'full_connected_prop']]
    housing_props.columns = ['independent', 'semi-connected', 'full-connected']
    
    plt.boxplot([housing_props['independent'].dropna(), 
                 housing_props['semi-connected'].dropna(), 
                 housing_props['full-connected'].dropna()],
                labels=['independent', 'semi-connected', 'full-connected'])
    plt.ylabel('Proportion')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return housing_summary

