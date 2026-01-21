import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv('birmingham_prices_real_2024.csv')
df['Postcode_Area'] = df['Postcode'].apply(lambda x: str(x).split(' ')[0])

# Comprehensive district mapping
district_names = {
    'B1': 'City Centre', 'B2': 'City Centre', 'B3': 'Jewellery Quarter',
    'B5': 'Digbeth', 'B7': 'Nechells/Aston', 'B13': 'Moseley',
    'B14': 'Kings Heath', 'B15': 'Edgbaston', 'B16': 'Ladywood',
    'B17': 'Harborne', 'B19': 'Lozells/Newtown', 'B29': 'Selly Oak',
    'B30': 'Stirchley', 'B42': 'Perry Barr', 'B66': 'Smethwick',
    'B72': 'Sutton Coldfield', 'B73': 'Sutton Coldfield',
    'B74': 'Sutton Coldfield', 'B75': 'Sutton Coldfield', 'B76': 'Sutton Coldfield'
}

df['District_Label'] = df['Postcode_Area'].apply(
    lambda x: f"{x} ({district_names.get(x, 'Other')})"
)

# Remove 'Other' to focus on identified Birmingham residential areas
df = df[df['District_Label'].str.contains('Other') == False]

# 2. Calculate Stats
# Raw stats
raw_stats = df.groupby('District_Label')['Price'].mean().reset_index()
raw_stats['Data_Type'] = 'Raw Data (Incl. Outliers)'

# Filtered stats (<= £1M)
filtered_df = df[df['Price'] <= 1000000].copy()
filtered_stats = filtered_df.groupby('District_Label')['Price'].mean().reset_index()
filtered_stats['Data_Type'] = 'Filtered Data (Mainstream <= £1M)'

# 3. CRITICAL: Sort by Filtered Price Descending
# We create a sorting order based on the filtered results
sort_order = filtered_stats.sort_values(by='Price', ascending=False)['District_Label'].tolist()

# Combine for plotting
comparison_df = pd.concat([raw_stats, filtered_stats])

# 4. Visualization
plt.figure(figsize=(14, 12))
sns.set_style("whitegrid")

# Use 'order' parameter to ensure sorting by Filtered Price
plot = sns.barplot(
    x='Price',
    y='District_Label',
    hue='Data_Type',
    data=comparison_df,
    order=sort_order,
    palette=['#34495e', '#e67e22'] # Dark Blue vs Orange
)

plt.title('Birmingham Market Value Ranking: Raw vs Filtered Comparison', fontsize=18, fontweight='bold')
plt.xlabel('Average Transaction Price (£)', fontsize=14)
plt.ylabel('District (Sorted by Filtered Market Price)', fontsize=14)
plt.legend(title='Price Calculation', loc='lower right')

# Add values on bars
for p in plot.patches:
    width = p.get_width()
    if width > 0:
        plt.text(width + 10000, p.get_y() + p.get_height()/2,
                 f'£{int(width):,}', va="center", fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('market_comparison_final.png')
plt.show()