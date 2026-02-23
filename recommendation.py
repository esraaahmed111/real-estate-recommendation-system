# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('ignore')


# Load data 
df = pd.read_csv('property_finder.csv')
pd.set_option("display.max_columns", None)

print("Dataset Shape:", df.shape)


# Data cleaning
df = df[df['bathrooms'] != 'None']
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce').astype('Int64')
df['bedrooms']  = pd.to_numeric(df['bedrooms'],  errors='coerce').astype('Int64')
df = df.dropna(subset=['bedrooms', 'bathrooms'])

df['down_payment_price'] = df['down_payment_price'].fillna(0)
df['district/compound']  = df['district/compound'].fillna('Unknown')
df['furnished'] = df['furnished'].astype(str).str.strip().str.capitalize()

df['completion_status'] = df['completion_status'].fillna('Unknown')
df.loc[df['completion_status'] == 'Unknown', 'completion_status'] = df['furnished'].map(
    {'Yes': 'Completed', 'Partly': 'Completed', 'No': 'Unknown'}
).fillna('Unknown')

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip().str.capitalize()

df.drop(columns=['url', 'id', 'location_full_name', 'has_view_360',
                 'amenity_names', 'payment_method', 'listed_date',
                 'offering_type'], inplace=True)

df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)


# Outliers
for col in ['price', 'size', 'bedrooms', 'bathrooms']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

df.reset_index(drop=True, inplace=True)
print(f"Clean dataset shape: {df.shape}")


# EDA
# Price distribution
filtered = df[df['price'] < df['price'].quantile(0.95)]
plt.figure(figsize=(12, 5))
sns.histplot(filtered['price'], bins=30, kde=True, color='steelblue', edgecolor='black')
plt.axvline(filtered['price'].mean(),   color='red',   linestyle='--', label=f'Mean:   {filtered["price"].mean():,.0f} EGP')
plt.axvline(filtered['price'].median(), color='green', linestyle='--', label=f'Median: {filtered["price"].median():,.0f} EGP')
plt.title('Property Price Distribution', fontsize=16)
plt.xlabel('Price (EGP)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Property type count
plt.figure(figsize=(12, 5))
order = df['property_type'].value_counts().index
sns.countplot(data=df, x='property_type', order=order, palette='viridis')
plt.title('Distribution of Property Types')
plt.xlabel('Property Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# price by property type
plt.figure(figsize=(12, 5))
med = df.groupby('property_type')['price'].median().sort_values(ascending=False)
ax = sns.barplot(x=med.index, y=med.values, palette='magma')
plt.title('Median Price by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Median Price (EGP)')
plt.xticks(rotation=45)
for i, v in enumerate(med.values):
    ax.text(i, v * 1.01, f'{v/1e6:.1f}M', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# price by city
plt.figure(figsize=(12, 5))
city_med = df.groupby('city')['price'].median().sort_values(ascending=False)
sns.barplot(x=city_med.index, y=city_med.values, palette='coolwarm')
plt.title('Median Price by City')
plt.xlabel('City')
plt.ylabel('Median Price (EGP)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Map visualization
fig = px.scatter_mapbox(
    df, lat='lat', lon='lon',
    color='price',
    size='size',
    color_continuous_scale='Viridis',
    size_max=15, zoom=6,
    hover_data={'price': True, 'property_type': True, 'furnished': True,
                'completion_status': True, 'city': True, 'bedrooms': True}
)
fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# Feature engineering 
df['price_per_sqm'] = df['price'] / df['size']
df['total_rooms']   = df['bedrooms'].astype(float) + df['bathrooms'].astype(float)


# Encode & Scale For KNN
model_df = df.copy()

categorical_cols = ['property_type', 'city', 'town', 'district/compound', 'furnished', 'completion_status']
numerical_cols   = ['price', 'lat', 'lon', 'bedrooms', 'bathrooms', 'size',
                    'down_payment_price', 'price_per_sqm', 'total_rooms']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    label_encoders[col] = le

scaler = MinMaxScaler()
model_df[numerical_cols] = scaler.fit_transform(model_df[numerical_cols])

prep_data = np.hstack([model_df[numerical_cols].to_numpy(),
                       model_df[categorical_cols].to_numpy()])


# KNN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
knn_model.fit(prep_data)

def recommend_by_index(index, n=6):
    """Recommend similar properties by DataFrame index."""
    input_vec = prep_data[index].reshape(1, -1)
    distances, indices = knn_model.kneighbors(input_vec, n_neighbors=n)
    recs = df.iloc[indices[0]].copy()
    recs['similarity_score'] = (1 - distances[0]).round(4)
    return recs[['property_type', 'city', 'town', 'price', 'size',
                 'bedrooms', 'bathrooms', 'furnished', 'similarity_score']]

# Example usage
print("\nRecommendations for property at index 7:")
print(recommend_by_index(7).to_string(index=False))


# 7. Dynamic filter + KNN on filtered results
def dynamic_filter(df):
    """Filter properties by user input, then run KNN on results."""
    print("\n Real Estate Search — press Enter to skip any field")

    city     = input(f"\nCity options: {sorted(df['city'].unique().tolist())}\nEnter city (or skip): ").strip().capitalize()
    town     = input(f"\nEnter town (or skip): ").strip().capitalize()
    prop_type= input(f"\nProperty type options: {sorted(df['property_type'].unique().tolist())}\nEnter type (or skip): ").strip().capitalize()
    furnished= input(f"\nFurnished? (Yes / No / Partly / skip): ").strip().capitalize()

    try:
        min_price = float(input("\nMin price EGP (or skip): ").strip() or 0) or None
    except:
        min_price = None

    try:
        max_price = float(input("Max price EGP (or skip): ").strip() or 0) or None
    except:
        max_price = None

    try:
        min_beds = int(input("Min bedrooms (or skip): ").strip() or 0) or None
    except:
        min_beds = None

    filtered = df.copy()
    if city:      filtered = filtered[filtered['city'] == city]
    if town:      filtered = filtered[filtered['town'] == town]
    if prop_type: filtered = filtered[filtered['property_type'] == prop_type]
    if furnished: filtered = filtered[filtered['furnished'] == furnished]
    if min_price: filtered = filtered[filtered['price'] >= min_price]
    if max_price: filtered = filtered[filtered['price'] <= max_price]
    if min_beds:  filtered = filtered[filtered['bedrooms'] >= min_beds]

    if filtered.empty:
        print("\n No properties match your criteria.")
        return pd.DataFrame()

    print(f"\n Found {len(filtered)} matching properties.")
    return filtered

def recommend_from_filter(filtered_df, n=5):
    """Run KNN within the filtered subset and show top similar properties."""
    if filtered_df.empty or len(filtered_df) < 2:
        print("Not enough results to recommend.")
        return

    indices = filtered_df.index.tolist()
    sub_data = prep_data[indices]

    sub_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    sub_knn.fit(sub_data)

    # Use the first result as the reference property
    ref = sub_data[0].reshape(1, -1)
    n_neighbors = min(n + 1, len(sub_data))
    distances, idx = sub_knn.kneighbors(ref, n_neighbors=n_neighbors)

    recs = filtered_df.iloc[idx[0][1:]].copy()
    recs['similarity_score'] = (1 - distances[0][1:]).round(4)

    print("\n Top Similar Properties:")
    print(recs[['property_type', 'city', 'town', 'price', 'size',
                'bedrooms', 'bathrooms', 'furnished', 'similarity_score']].to_string(index=False))
    return recs


# Run the search
filtered_results = dynamic_filter(df)
if not filtered_results.empty:
    recommend_from_filter(filtered_results)
