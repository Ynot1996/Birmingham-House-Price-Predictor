import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('birmingham_prices_real_2024.csv')
df['Postcode_Area'] = df['Postcode'].apply(lambda x: str(x).split(' ')[0])

# 1. Expanded mapping to remove "Other" as much as possible
district_names = {
    'B1': 'City Centre',
    'B2': 'City Centre',
    'B3': 'Jewellery Quarter',
    'B5': 'Digbeth',
    'B7': 'Nechells/Aston',
    'B13': 'Moseley',
    'B14': 'Kings Heath',
    'B15': 'Edgbaston',
    'B16': 'Ladywood',
    'B17': 'Harborne',
    'B19': 'Lozells/Newtown',
    'B29': 'Selly Oak',
    'B30': 'Stirchley',
    'B42': 'Perry Barr',
    'B66': 'Smethwick',
    'B72': 'Sutton Coldfield',
    'B73': 'Sutton Coldfield',
    'B74': 'Sutton Coldfield',
    'B75': 'Sutton Coldfield',
    'B76': 'Sutton Coldfield'
}

df['District_Label'] = df['Postcode_Area'].apply(
    lambda x: f"{x} ({district_names.get(x, 'Birmingham Area')})"
)

# 2. Re-calculate and remove 'nan' rows if any
stats = df.dropna(subset=['Postcode_Area'])
stats = stats.groupby('District_Label')['Price'].mean().sort_values(ascending=False).head(15).reset_index()

# 3. Visualization with fixed warning
plt.figure(figsize=(14, 10))
sns.set_style("whitegrid")

# Fixed the warning by adding hue='District_Label' and legend=False
plot = sns.barplot(
    x='Price',
    y='District_Label',
    data=stats,
    palette='magma',
    hue='District_Label',
    legend=False
)

plt.title('Birmingham Real Estate Market Analysis (2024 Average)', fontsize=18, fontweight='bold')
plt.xlabel('Average Transaction Price (£)', fontsize=14)
plt.ylabel('Location', fontsize=14)

for p in plot.patches:
    width = p.get_width()
    plt.text(width + 5000, p.get_y() + p.get_height()/2, f'£{int(width):,}', va="center", fontweight='bold')

plt.tight_layout()
plt.savefig('birmingham_final_market_map.png')
plt.show()