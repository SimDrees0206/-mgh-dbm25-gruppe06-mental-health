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
     

Selektiere nur numerische Spalten für die statistische Auswertung
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns

summary_data = {}

for col in numeric_cols:
    col_data = df_filtered[col].dropna()
    n = len(col_data)

    if n == 0:
        continue

    mean_val = col_data.mean()
    median_val = col_data.median()

    # Modus berechnen
    mode_result = col_data.mode()
    mode_val = mode_result.iloc[0] if not mode_result.empty else np.nan

    std_val = col_data.std()
    min_val = col_data.min()
    max_val = col_data.max()
    range_val = max_val - min_val
    sum_val = col_data.sum()

    # 95% Konfidenzintervall / Konfidenzniveau (Margin of Error)
    if n > 1 and std_val > 0:
        sem = stats.sem(col_data)  # Standardfehler des Mittelwerts
        confidence_level = 0.95
        # t-Verteilung für kleinere Stichproben, Normalverteilung für große
        margin_of_error = sem * stats.t.ppf((1 + confidence_level) / 2, n - 1)
    else:
        margin_of_error = np.nan

    summary_data[col] = {
        "Mittelwert": mean_val,
        "Median": median_val,
        "Modus": mode_val,
        "Standardabweichung": std_val,
        "Wertebereich": range_val,
        "Minimum": min_val,
        "Maximum": max_val,
        "Summe": sum_val,
        "Anzahl": n,
        "Konfidenzniveau (95%)": margin_of_error
    }

In ein DataFrame für eine übersichtliche Darstellung umwandeln
df_summary = pd.DataFrame(summary_data).T
display(df_summary)


import matplotlib.pyplot as plt
import seaborn as sns

Bestimmung der exakten Spaltennamen im Datensatz
Wir suchen nach Spalten, die 'anxiety', 'headache' und 'loneliness' enthalten
anxiety_col = [col for col in df_filtered.columns if 'anxiety' in col.lower()][0]
headache_col = [col for col in df_filtered.columns if 'headache' in col.lower()][0]
loneliness_col = [col for col in df_filtered.columns if 'loneliness' in col.lower()][0]
gender_col = [col for col in df_filtered.columns if 'gender' in col.lower()][0]

1. Tabellarische Übersicht (Mittelwerte und Standardabweichungen nach Geschlecht)
grouped_stats = df_filtered.groupby(gender_col)[[anxiety_col, headache_col, loneliness_col]].agg(['mean', 'std']).round(3)
print("Tabellarischer Vergleich (Mittelwert & Standardabweichung):")
display(grouped_stats)

2. Daten für das Diagramm vorbereiten
df_melted = df_filtered.melt(id_vars=[gender_col], value_vars=[anxiety_col, headache_col, loneliness_col],
                             var_name='Indikator', value_name='Ausprägung')

3. Diagramm erstellen
plt.figure(figsize=(12, 6))
sns.barplot(x='Indikator', y='Ausprägung', hue=gender_col, data=df_melted, palette='muted', errorbar='ci')
plt.title('Vergleich von Stressindikatoren zwischen weiblichen und männlichen Studierenden (95% CI)')
plt.ylabel('Durchschnittliche Ausprägung')
plt.xlabel('Stressindikator')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


Identifikation relevanter Spalten
Häuslich/Sozial: family_conflicts, peer_pressure
Universitär: academic_conflicts, study_load, class_attendance
Allgemeines Stressempfinden: stress_experience
stress_exp_col = [col for col in df_filtered.columns if 'stress_experience' in col.lower()][0]

Wir wählen repräsentative Variablen aus
home_env_cols = [col for col in df_filtered.columns if 'family' in col.lower() or 'home' in col.lower()]
uni_env_cols = [col for col in df_filtered.columns if 'academic' in col.lower() or 'study' in col.lower() or 'class' in col.lower()]

all_env_cols = home_env_cols + uni_env_cols

1. Korrelationsmatrix mit dem allgemeinen Stressempfinden
correlations = df_filtered[all_env_cols + [stress_exp_col]].corr()[stress_exp_col].drop(stress_exp_col).to_frame()
correlations.columns = ['Korrelation mit allgemeinem Stress']
correlations = correlations.sort_values(by='Korrelation mit allgemeinem Stress', ascending=False)

print("Einfluss der Umfelder auf das Stressempfinden (Korrelationskoeffizienten):")
display(correlations.round(3))

2. Diagramm erstellen
plt.figure(figsize=(10, 6))
sns.heatmap(df_filtered[all_env_cols + [stress_exp_col]].corr()[[stress_exp_col]].sort_values(by=stress_exp_col, ascending=False),
            annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".3f")
plt.title('Korrelation von häuslichen & universitären Faktoren mit allgemeinem Stress')
plt.tight_layout()
plt.show()


