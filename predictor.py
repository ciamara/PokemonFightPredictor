import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

pd.set_option('display.max_rows', None)

# Data preparation
pokemon = pd.read_csv('data/pokemon.csv')
combats = pd.read_csv('data/combatsUpdated.csv')

numeric_features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
features = numeric_features + ['Legendary', 'Type 1', 'Type 2']
pokemon_sub = pokemon[['#'] + features]

df = combats.merge(pokemon_sub, left_on='First_pokemon', right_on='#', how='left')
df = df.merge(pokemon_sub, left_on='Second_pokemon', right_on='#', how='left', suffixes=('_1', '_2'))

# Base analysis
for col in numeric_features:
    df[f'{col}_diff'] = df[f'{col}_1'] - df[f'{col}_2']

diff_cols = [f'{col}_diff' for col in numeric_features]
base_features = diff_cols + ['Legendary_1', 'Legendary_2']

df['Legendary_1'] = df['Legendary_1'].astype(int)
df['Legendary_2'] = df['Legendary_2'].astype(int)

X_base = df[base_features].fillna(0)
y = df['First_win'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

all_results = {}

# Base models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# -------------------------- BASE MODELS --------------------------

print("\n1. Base Models")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    all_results[f"{name} (Base)"] = acc
    print(f"{name}: {acc:.4f}")


rf = models['Random Forest']
importancesRF = pd.Series(rf.feature_importances_, index=base_features).sort_values(ascending=False)
print("\nFeature Importances (Random Forest - Base)")
print(importancesRF)

gb = models['Gradient Boosting']
importancesGB = pd.Series(gb.feature_importances_, index=base_features).sort_values(ascending=False)
print("\nFeature Importances (Gradient Boosting - Base)")
print(importancesGB)

lr = models['Logistic Regression']
importancesLR = pd.Series(np.abs(lr.coef_[0]), index=base_features).sort_values(ascending=False)
print("\nFeature Importances (Logistic Regression - Base)")
print(importancesLR)

# -------------------------- ONE HOT ENCODING (TYPES) --------------------------

print("\n2. Random Forest with One-Hot Encoding on the Types")

type_columns = ['Type 1_1', 'Type 2_1', 'Type 1_2', 'Type 2_2']
df[type_columns] = df[type_columns].fillna('None')

df_ohe = pd.get_dummies(df[type_columns])

X_ohe = pd.concat([X_base, df_ohe], axis=1)
X_train_ohe, X_test_ohe, _, _ = train_test_split(X_ohe, y, test_size=0.2, random_state=42)

rf_ohe = RandomForestClassifier(n_estimators=100, random_state=42)
rf_ohe.fit(X_train_ohe, y_train)

acc_ohe = accuracy_score(y_test, rf_ohe.predict(X_test_ohe))
all_results['Random Forest (OHE)'] = acc_ohe
print(f"Random Forest (with OHE): {acc_ohe:.4f}")

importancesOHE = pd.Series(rf_ohe.feature_importances_, index=X_ohe.columns).sort_values(ascending=False)
print("\nFeature Importances (Random Forest with OHE)")
print(importancesOHE)

# -------------------------- CAT BOOST --------------------------

print("\n3. CatBoost")

X_catboost = df[base_features + type_columns]
X_train_cb, X_test_cb, _, _ = train_test_split(X_catboost, y, test_size=0.2, random_state=42)

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    cat_features=type_columns,
    verbose=0,
    random_state=42
)

cat_model.fit(X_train_cb, y_train)

acc_cb = accuracy_score(y_test, cat_model.predict(X_test_cb))
all_results['CatBoost'] = acc_cb
print(f"CatBoost: {acc_cb:.4f}")

importancesCB = pd.Series(cat_model.feature_importances_, index=X_catboost.columns).sort_values(ascending=False)
print("\nFeature Importances (CatBoost)")
print(importancesCB)

print("\n==================================================")
print("ACCURACY SUMMARY")
print("==================================================")
for model_name, acc in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:<30} | {acc:.4f}")
print("==================================================")