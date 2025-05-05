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

# Calculate correlation coefficient
correlation = merged_data['Y2013'].corr(merged_data['Score'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_data['Y2013'], merged_data['Score'])

# Set non-interactive backend if in non-interactive environment
plt.switch_backend('Agg')

# Create scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    merged_data['Y2013'], merged_data['Score'],
    alpha=0.7, c='blue', edgecolors='black'
)

# Regression line
x = np.array(merged_data['Y2013'])
regression_line = intercept + slope * x
plt.plot(x, regression_line, 'r', label=f'Regression line (corr = {correlation:.2f})')

# Labels and title
plt.xlabel('Total Human Food Production in 2013 (Measured in 1000 tons)', fontsize=14)
plt.ylabel('Happiness Index (0–10)', fontsize=14)
plt.title('Correlation between Human Food Production and Happiness Index by Country', fontsize=16)

# Grid and legend
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
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
