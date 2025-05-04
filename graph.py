import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load the two datasets with the correct file names
food_production = pd.read_csv('FAO.csv', encoding='latin1')
happiness_index = pd.read_csv('2019.csv')

# Merge datasets with the correct column names for countries
merged_data = pd.merge(food_production, happiness_index, 
                      left_on='Area', right_on='Country or region', 
                      how='inner')

print(f"Successfully matched {len(merged_data)} countries between datasets")

food_production_col = 'Y2013'  # Replace with actual column name for food production
happiness_index_col = 'Score'      # Replace with actual column name for happiness index

# Calculate correlation coefficient
correlation = merged_data[food_production_col].corr(merged_data[happiness_index_col])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_data[food_production_col], merged_data[happiness_index_col])


# Create scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(merged_data[food_production_col], merged_data[happiness_index_col], 
                     alpha=0.7, c='blue', edgecolors='black')


# Add regression line
x = np.array(merged_data[food_production_col])
plt.plot(x, intercept + slope*x, 'r', label=f'Correlation: {r_value:.2f}')

# Add labels and title
plt.xlabel('Food Production', fontsize=14)
plt.ylabel('Happiness Index (0-10)', fontsize=14)
plt.title('Correlation between Food Production and Happiness Index by Country', fontsize=16)

# Add correlation information
plt.annotate(f'Correlation Coefficient: {correlation:.2f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

# Add country names as annotations
n_countries = len(merged_data)
step = max(1, n_countries // 15)  # Annotate at most 15 countries

for i in range(0, n_countries, step):
    plt.annotate(merged_data['Area'].iloc[i], 
                (merged_data[food_production_col].iloc[i], merged_data[happiness_index_col].iloc[i]),
                fontsize=8, xytext=(5, 5), textcoords='offset points')

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('food_production_vs_happiness.png', dpi=300)

# Show the plot
plt.show()

# Print statistical summary
print(f"\nStatistical Summary:")
print(f"Correlation coefficient: {correlation:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")