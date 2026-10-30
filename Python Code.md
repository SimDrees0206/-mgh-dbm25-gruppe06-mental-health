import pandas as pd
import numpy as np
import scipy.stats as stats

Direktes Laden der CSV-Datei von GitHub
file_path = 'https://raw.githubusercontent.com/SimDrees0206/-mgh-dbm25-gruppe06-mental-health/refs/heads/main/mental_health_raw_data.csv'

Daten laden
try:
    df = pd.read_csv(file_path)
    print("Daten erfolgreich direkt von GitHub geladen!")
except Exception as e:
    print(f"Fehler beim Laden der Datei von GitHub: {e}")

Filtern auf die Altersgruppe 18-22 (inklusive 18 und 22)
age_column = [col for col in df.columns if 'age' in col.lower()][0]
df_filtered = df[(df[age_column] >= 18) & (df[age_column] <= 22)].copy()

print(f"Anzahl der Datensätze nach Filterung (18-22 Jahre): {len(df_filtered)}")
display(df_filtered.head())


Identifizieren der Gender-Spalte
gender_column = [col for col in df_filtered.columns if 'gender' in col.lower()][0]

Mapping anwenden: 0 zu 'M' und 1 zu 'F'
Wir konvertieren die Spalte zuerst in numerische Werte, falls sie als String vorliegen
df_filtered[gender_column] = pd.to_numeric(df_filtered[gender_column], errors='coerce')
df_filtered[gender_column] = df_filtered[gender_column].map({0: 'M', 1: 'F'})

print("Verteilung des Geschlechts nach der Anpassung:")
print(df_filtered[gender_column].value_counts(dropna=False))
display(df_filtered.head())
     
