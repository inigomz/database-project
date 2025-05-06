import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the datasets
emissions_df = pd.read_csv('co2_emissions_kt_by_country.csv')
fao_df = pd.read_csv('FAO.csv', encoding='latin1')

# Filter out India from both datasets
emissions_df = emissions_df[emissions_df['country_name'] != 'India']
fao_df = fao_df[fao_df['Area'] != 'India']

# Filter emissions data for the year 2013
emissions_2013 = emissions_df[emissions_df['year'] == 2013]
emissions_by_country = emissions_2013[['country_name', 'value']].copy()
emissions_by_country.columns = ['Country', 'CO2_Emissions_kt']

# Filter food production data (Element Code 5142) and sum by country
food_production = fao_df[fao_df['Element Code'] == 5142][['Area', 'Y2013']].copy()
food_production.columns = ['Country', 'Food_Production']
food_by_country = food_production.groupby('Country').sum().reset_index()

# Merge the datasets
merged_data = pd.merge(emissions_by_country, food_by_country, on='Country', how='inner')

# --- Regression calculation ---
x = merged_data['Food_Production']
y = merged_data['CO2_Emissions_kt']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
regression_line = intercept + slope * x

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
scatter = ax.scatter(
    x,
    y,
    alpha=0.6,
    c='blue',
    edgecolors='black',
    linewidths=1
)

# Regression line
ax.plot(x, regression_line, 'r', label=f'Regression line (corr = {r_value:.2f})')

# Labels and formatting
ax.set_title('Carbon Emissions vs Food Production by Country (2013)', fontsize=16)
ax.set_xlabel('Food Production (units)', fontsize=12)
ax.set_ylabel('CO2 Emissions (kilotons)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.ticklabel_format(style='plain', axis='both')
ax.legend(loc='upper left')

# Adjust layout
fig.tight_layout()

# Save and show
fig.savefig('emissions_vs_food_production_2013_no_india.png', dpi=300)
plt.show()

# Print summary
print(f"Total countries in the analysis (excluding India): {len(merged_data)}")
print(f"Correlation coefficient (r): {r_value:.2f}")
print(f"R-squared: {r_value**2:.2f}")
print("Top 5 countries by CO2 emissions:")
print(merged_data.nlargest(5, 'CO2_Emissions_kt')[['Country', 'CO2_Emissions_kt', 'Food_Production']])
