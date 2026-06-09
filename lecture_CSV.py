import pandas as pd

# In your while loop, before messages.append:
while True:
    user_input = input("> ")
    if user_input.startswith("csv "):
        path = user_input[4:].strip()
        try:
            df = pd.read_csv(path)
            result = f"Colonnes: {list(df.columns)}\n{df.head(5).to_string()}"
            print(f"\nCSV:\n{result}")
        except Exception as e:
            print(f"\nErreur: {e}")
        continue