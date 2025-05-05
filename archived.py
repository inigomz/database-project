import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# Load food production data with proper encoding
try:
    food_production = pd.read_csv('FAO.csv', encoding='latin1')
    print("Successfully read FAO.csv")
except Exception as e:
    print(f"Error reading FAO.csv: {e}")

# Load happiness index data
try:
    happiness_index = pd.read_csv('2019.csv')
    print("Successfully read 2019.csv")
except Exception as e:
    print(f"Error reading 2019.csv: {e}")

# Check if necessary columns exist in food_production
if 'Area' not in food_production.columns or 'Y2013' not in food_production.columns or 'Element Code' not in food_production.columns:
    print(f"Warning: Required columns not found in FAO.csv. Available columns are:")
    print(food_production.columns.tolist())

# Filter for only food production entries (Element Code = 5142) and then aggregate by country
food_only = food_production[food_production['Element Code'] == 5142]
print(f"Filtered {len(food_only)} food production entries (Element Code = 5142)")

# Aggregate food production by summing Y2013 values for each country
total_food_by_country = food_only.groupby('Area')['Y2013'].sum().reset_index()
print(f"Aggregated food production data for {len(total_food_by_country)} countries")

# Remove India from the dataset as it's an outlier
total_food_by_country = total_food_by_country[total_food_by_country['Area'] != 'India']
print(f"Removed India from dataset. Remaining countries: {len(total_food_by_country)}")

# Check if necessary columns exist in happiness_index
if 'Country or region' not in happiness_index.columns or 'Score' not in happiness_index.columns:
    print(f"Warning: Required columns not found in 2019.csv. Available columns are:")
    print(happiness_index.columns.tolist())

# Merge the datasets based on country names
merged_data = pd.merge(total_food_by_country, happiness_index, 
                       left_on='Area', right_on='Country or region', 
                       how='inner')

print(f"Successfully matched {len(merged_data)} countries between datasets")

# Create a log-transformed version of the food production data
# Add a small constant (1) to handle any zeros
merged_data['Log_Food_Production'] = np.log10(merged_data['Y2013'] + 1)

# Calculate correlation for both original and log-transformed data
correlation_original = merged_data['Y2013'].corr(merged_data['Score'])
correlation_log = merged_data['Log_Food_Production'].corr(merged_data['Score'])

# Calculate regression statistics for both
slope_orig, intercept_orig, r_value_orig, p_value_orig, std_err_orig = stats.linregress(
    merged_data['Y2013'], merged_data['Score'])

slope_log, intercept_log, r_value_log, p_value_log, std_err_log = stats.linregress(
    merged_data['Log_Food_Production'], merged_data['Score'])

# Set non-interactive backend if in non-interactive environment
plt.switch_backend('Agg')

# Create figure with two subplots - original and log-transformed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Original data
ax1.scatter(merged_data['Y2013'], merged_data['Score'], 
           alpha=0.7, c='blue', edgecolors='black')

# Add regression line to original data
x_orig = np.array(merged_data['Y2013'])
ax1.plot(x_orig, intercept_orig + slope_orig*x_orig, 'r', label=f'Regression line')

# Set x-axis limit for original plot
max_x_value = merged_data['Y2013'].max()
ax1.set_xlim(0, max_x_value * 1.1)

# Add labels and title to original plot
ax1.set_xlabel('Total Human Food Production (2013)', fontsize=12)
ax1.set_ylabel('Happiness Index (0-10)', fontsize=12)
ax1.set_title('Original Scale', fontsize=14)

# Add correlation information to original plot
ax1.annotate(f'Correlation: {correlation_original:.2f}\nR²: {r_value_orig**2:.2f}', 
           xy=(0.05, 0.95), xycoords='axes fraction', 
           fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

# # Add country names to original plot (selected points)
# n_countries = len(merged_data)
# step = max(1, n_countries // 8)
# for i in range(0, n_countries, step):
#     ax1.annotate(merged_data['Area'].iloc[i], 
#                (merged_data['Y2013'].iloc[i], merged_data['Score'].iloc[i]),
#                fontsize=8, xytext=(5, 5), textcoords='offset points')

ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Log-transformed data
ax2.scatter(merged_data['Log_Food_Production'], merged_data['Score'], 
           alpha=0.7, c='green', edgecolors='black')

# Add regression line to log-transformed data
x_log = np.array(merged_data['Log_Food_Production'])
ax2.plot(x_log, intercept_log + slope_log*x_log, 'r', label=f'Regression line')

# Add labels and title to log-transformed plot
ax2.set_xlabel('Log10(Total Human Food Production + 1)', fontsize=12)
ax2.set_ylabel('Happiness Index (0-10)', fontsize=12)
ax2.set_title('Log-Transformed Scale (Better Distribution)', fontsize=14)

# Add correlation information to log-transformed plot
ax2.annotate(f'Correlation: {correlation_log:.2f}\nR²: {r_value_log**2:.2f}', 
           xy=(0.05, 0.95), xycoords='axes fraction', 
           fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

# # Add country names to log-transformed plot
# for i in range(0, n_countries, step):
#     ax2.annotate(merged_data['Area'].iloc[i], 
#                (merged_data['Log_Food_Production'].iloc[i], merged_data['Score'].iloc[i]),
#                fontsize=8, xytext=(5, 5), textcoords='offset points')

ax2.grid(True, alpha=0.3)
ax2.legend()

# Add overall title
plt.suptitle('Correlation between Human Food Production and Happiness Index by Country', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the overall title

# Save the figure
output_file = 'food_production_vs_happiness_comparison.png'
plt.savefig(output_file, dpi=300)
print(f"Plot saved as '{output_file}' in {os.getcwd()}")

# Try to show the plot, but don't error if can't display
try:
    plt.show()
except Exception as e:
    print(f"Note: Plot couldn't be displayed interactively but was saved to file. Error: {e}")

# Print statistical summary
print(f"\nStatistical Summary (Original Data):")
print(f"Correlation coefficient: {correlation_original:.4f}")
print(f"R-squared: {r_value_orig**2:.4f}")
print(f"p-value: {p_value_orig:.4f}")

print(f"\nStatistical Summary (Log-Transformed Data):")
print(f"Correlation coefficient: {correlation_log:.4f}")
print(f"R-squared: {r_value_log**2:.4f}")
print(f"p-value: {p_value_log:.4f}")

# Save the merged data to CSV for further analysis if needed
merged_data.to_csv('food_production_happiness_data_with_log.csv', index=False)
print("Merged data saved to 'food_production_happiness_data_with_log.csv'")