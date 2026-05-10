import pandas as pd


combats = pd.read_csv("data/combatsUpdated.csv")

combats['First_win'] = combats['First_pokemon'] == combats['Winner']

total_rows = len(combats)
target_count = total_rows // 2  # 25000
current_false_count = combats['First_win'].value_counts().get(False, 0)

rows_to_swap = current_false_count - target_count 

indices_to_swap = combats[combats['First_win'] == False].sample(n=rows_to_swap, random_state=42).index

combats.loc[indices_to_swap, ['First_pokemon', 'Second_pokemon']] = combats.loc[indices_to_swap, ['Second_pokemon', 'First_pokemon']].values

combats.loc[indices_to_swap, 'First_win'] = True

count_first_win = combats['First_win'].value_counts().get(True, 0)
count_second_win = combats['First_win'].value_counts().get(False, 0)

print("First wins: ", count_first_win)
print("Second wins: ", count_second_win)

combats.to_csv("data/combatsUpdated.csv", index=False)