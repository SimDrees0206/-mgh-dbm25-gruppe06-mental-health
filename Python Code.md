import pandas as pd
import numpy as np
import scipy.stats as stats

# Direktes Laden der CSV-Datei von GitHub
file_path = 'https://raw.githubusercontent.com/SimDrees0206/-mgh-dbm25-gruppe06-mental-health/refs/heads/main/mental_health_raw_data.csv'

# Daten laden
try:
    df = pd.read_csv(file_path)
    print("Daten erfolgreich direkt von GitHub geladen!")
except Exception as e:
    print(f"Fehler beim Laden der Datei von GitHub: {e}")

# Filtern auf die Altersgruppe 18-22 (inklusive 18 und 22)
age_column = [col for col in df.columns if 'age' in col.lower()][0]
df_filtered = df[(df[age_column] >= 18) & (df[age_column] <= 22)].copy()

print(f"Anzahl der Datensätze nach Filterung (18-22 Jahre): {len(df_filtered)}")
display(df_filtered.head())


