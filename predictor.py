import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


pokemon = pd.read_csv('data/pokemon.csv')
combats = pd.read_csv('data/combatsUpdated.csv')

numeric_features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
features = numeric_features + ['Legendary']
pokemon_sub = pokemon[['#'] + features]


df = combats.merge(pokemon_sub, left_on='First_pokemon', right_on='#', how='left')

df = df.merge(pokemon_sub, left_on='Second_pokemon', right_on='#', how='left', suffixes=('_1', '_2'))


for col in numeric_features:
    df[f'{col}_diff'] = df[f'{col}_1'] - df[f'{col}_2']


diff_cols = [f'{col}_diff' for col in numeric_features]
final_features = diff_cols + ['Legendary_1', 'Legendary_2']


df['Legendary_1'] = df['Legendary_1'].astype(int)
df['Legendary_2'] = df['Legendary_2'].astype(int)

X = df[final_features].fillna(0)
y = df['First_win'].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}


results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)


rf = models['Random Forest']
importancesRF = pd.Series(rf.feature_importances_, index=final_features).sort_values(ascending=False)

gb = models['Gradient Boosting']
importancesGB = pd.Series(gb.feature_importances_, index=final_features).sort_values(ascending=False)

lr = models['Logistic Regression']
importancesLR = pd.Series(np.abs(lr.coef_[0]), index=final_features).sort_values(ascending=False)


print("--- Model Accuracies ---")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

print("\n--- Feature Importances (Random Forest) ---")
print(importancesRF)

print("\n--- Feature Importances (Gradient Boosting) ---")
print(importancesGB)

print("\n--- Feature Importances (Logistic Regression - Absolute Coefficients) ---")
print(importancesLR)