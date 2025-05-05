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

# # Remove India from the dataset as it's an outlier
# total_food_by_country = total_food_by_country[total_food_by_country['Area'] != 'India']
# print(f"Removed India from dataset. Remaining countries: {len(total_food_by_country)}")

# Check if necessary columns exist in happiness_index
if 'Country or region' not in happiness_index.columns or 'Score' not in happiness_index.columns:
    print(f"Warning: Required columns not found in 2019.csv. Available columns are:")
    print(happiness_index.columns.tolist())

# Merge the datasets based on country names
merged_data = pd.merge(total_food_by_country, happiness_index, 
                       left_on='Area', right_on='Country or region', 
                       how='inner')

print(f"Successfully matched {len(merged_data)} countries between datasets")

# Calculate correlation coefficient
correlation = merged_data['Y2013'].corr(merged_data['Score'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_data['Y2013'], merged_data['Score'])

# Set non-interactive backend if in non-interactive environment
plt.switch_backend('Agg')

# Create scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(merged_data['Y2013'], merged_data['Score'], 
                     alpha=0.7, c='blue', edgecolors='black')

# Add regression line
x = np.array(merged_data['Y2013'])
plt.plot(x, intercept + slope*x, 'r', label=f'Regression line')

# # Set x-axis limit to be approximately 1/3 of the maximum value
# max_x_value = merged_data['Y2013'].max()
# plt.xlim(0, max_x_value * 1.1)  # Add a little padding (10%)

# Add labels and title
plt.xlabel('Total Human Food Production in 2013 (Measured in 1000 tons)', fontsize=14)
plt.ylabel('Happiness Index (0-10)', fontsize=14)
plt.title('Correlation between Human Food Production and Happiness Index by Country', fontsize=16)

# Add correlation information
plt.annotate(f'Correlation: {correlation:.2f}\nR²: {r_value**2:.2f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

# Add country names as annotations (for clarity, only add for some points)
n_countries = len(merged_data)

for i in range(0, n_countries):
    plt.annotate(merged_data['Area'].iloc[i], 
                (merged_data['Y2013'].iloc[i], merged_data['Score'].iloc[i]),
                fontsize=8, xytext=(5, 5), textcoords='offset points')

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the figure
output_file = 'food_production_vs_happiness_no_india.png'
plt.savefig(output_file, dpi=300)
print(f"Plot saved as '{output_file}' in {os.getcwd()}")

# Try to show the plot, but don't error if can't display
try:
    plt.show()
except Exception as e:
    print(f"Note: Plot couldn't be displayed interactively but was saved to file. Error: {e}")

# Print statistical summary
print(f"\nStatistical Summary:")
print(f"Correlation coefficient: {correlation:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

# Save the merged data to CSV for further analysis if needed
merged_data.to_csv('food_production_happiness_data_no_india.csv', index=False)
print("Merged data saved to 'food_production_happiness_data_no_india.csv'")