# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')


# Load Data
df = pd.read_csv('property_finder.csv')
pd.set_option("display.max_columns", None)

print("Dataset Shape:", df.shape)
print(df.info())
print(df.describe())


# Data Cleaning
df = df[df['bathrooms'] != 'None']
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce').astype('Int64')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').astype('Int64')
df = df.dropna(subset=['bedrooms', 'bathrooms'])

df['down_payment_price'] = df['down_payment_price'].fillna(0)
df['district/compound'] = df['district/compound'].fillna('Unknown')

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

print(f"\nAfter cleaning: {df.shape}")
print(df.isnull().sum())


# Outliers
for col in ['price', 'size', 'bedrooms', 'bathrooms']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

print(f"After outlier removal: {df.shape}")


# EDA
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, col in zip(axes.flatten(), ['price', 'size', 'bedrooms', 'bathrooms']):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f'Box Plot of {col.capitalize()} (After Outlier Removal)')
plt.tight_layout()
plt.show()

# Price distribution
plt.figure(figsize=(12, 5))
sns.histplot(df['price'], bins=40, kde=True, color='steelblue', edgecolor='black')
plt.axvline(df['price'].mean(), color='red', linestyle='--', label=f'Mean: {df["price"].mean():,.0f} EGP')
plt.axvline(df['price'].median(), color='green', linestyle='--', label=f'Median: {df["price"].median():,.0f} EGP')
plt.title('Property Price Distribution')
plt.xlabel('Price (EGP)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Price by city
plt.figure(figsize=(12, 5))
city_med = df.groupby('city')['price'].median().sort_values(ascending=False)
sns.barplot(x=city_med.index, y=city_med.values, palette='viridis')
plt.title('Median Price by City')
plt.xlabel('City')
plt.ylabel('Median Price (EGP)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Price by property type
plt.figure(figsize=(12, 5))
type_med = df.groupby('property_type')['price'].median().sort_values(ascending=False)
sns.barplot(x=type_med.index, y=type_med.values, palette='magma')
plt.title('Median Price by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Median Price (EGP)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# Feature engineering
df['price_per_sqm'] = df['price'] / df['size']
df['total_rooms'] = df['bedrooms'].astype(float) + df['bathrooms'].astype(float)


# Prepare features 
categorical_cols = ['property_type', 'city', 'town', 'district/compound',
                    'completion_status', 'furnished']
numerical_cols   = ['lat', 'lon', 'bedrooms', 'bathrooms', 'size',
                    'down_payment_price', 'total_rooms']

X = df[categorical_cols + numerical_cols]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
column_trans = ColumnTransformer(transformers=[
    ('num', MinMaxScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])


# Bagging
base_estimator = DecisionTreeRegressor(max_depth=None, min_samples_leaf=2, random_state=42)

bag = BaggingRegressor(
    estimator=base_estimator,
    n_estimators=200,   # reduced from 500 for speed; increase if needed
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

bag_pipeline = Pipeline([('prep', column_trans), ('bag', bag)])
bag_pipeline.fit(X_train, y_train)

bag_oob = bag_pipeline.named_steps['bag'].oob_score_
bag_train_r2 = r2_score(y_train, bag_pipeline.predict(X_train))
bag_test_r2  = r2_score(y_test,  bag_pipeline.predict(X_test))
bag_mae      = mean_absolute_error(y_test, bag_pipeline.predict(X_test))

print("\n===== Bagging Regressor =====")
print(f"OOB R²       : {bag_oob:.4f}")
print(f"Train R²     : {bag_train_r2:.4f}")
print(f"Test R²      : {bag_test_r2:.4f}")
print(f"MAE on test  : {bag_mae:,.0f} EGP")


# Histgradiantboosting
X_train_trans = column_trans.fit_transform(X_train)
X_test_trans  = column_trans.transform(X_test)

boost_model = HistGradientBoostingRegressor(
    learning_rate=0.05,      # lowered for better generalization
    max_iter=1000,
    max_depth=8,             # slightly less deep to reduce overfitting
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

boost_model.fit(X_train_trans, y_train)

y_train_pred = boost_model.predict(X_train_trans)
y_test_pred  = boost_model.predict(X_test_trans)

print("\n===== HistGradientBoosting Regressor =====")
print(f"Train R²  : {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R²   : {r2_score(y_test,  y_test_pred):.4f}")
print(f"MAE       : {mean_absolute_error(y_test, y_test_pred):,.0f} EGP")
print(f"RMSE      : {np.sqrt(mean_squared_error(y_test, y_test_pred)):,.0f} EGP")


# Model comparison
comparison = pd.DataFrame({
    'Model'  : ['Bagging Regressor', 'HistGradientBoosting'],
    'Train R²': [round(bag_train_r2, 4), round(r2_score(y_train, y_train_pred), 4)],
    'Test R²' : [round(bag_test_r2,  4), round(r2_score(y_test,  y_test_pred),  4)],
    'MAE (EGP)': [f"{bag_mae:,.0f}", f"{mean_absolute_error(y_test, y_test_pred):,.0f}"]
})
print("\n===== Model Comparison =====")
print(comparison.to_string(index=False))


# Actual VS Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3, color='steelblue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Price (EGP)")
plt.ylabel("Predicted Price (EGP)")
plt.title("Actual vs Predicted Prices — HistGradientBoosting")
plt.legend()
plt.tight_layout()
plt.show()

# Residuals plot
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=40, kde=True, color='coral')
plt.axvline(0, color='black', linestyle='--')
plt.title('Residuals Distribution')
plt.xlabel('Residual (Actual - Predicted)')
plt.tight_layout()
plt.show()


# Top 10
results = pd.DataFrame({
    'Actual Price (EGP)'   : y_test.values,
    'Predicted Price (EGP)': y_test_pred.round(0)
})
results['Error (EGP)'] = (results['Actual Price (EGP)'] - results['Predicted Price (EGP)']).abs()
results['Error %'] = (results['Error (EGP)'] / results['Actual Price (EGP)'] * 100).round(2)

print("\nTop 10 Predictions:")
print(results.head(10).to_string(index=False))
