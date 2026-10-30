import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

file_path = 'https://raw.githubusercontent.com/SimDrees0206/-mgh-dbm25-gruppe06-mental-health/refs/heads/main/mental_health_raw_data.csv'

try:
    df = pd.read_csv(file_path)
    print("Daten erfolgreich direkt von GitHub geladen!")
except Exception as e:
    print(f"Fehler beim Laden der Datei von GitHub: {e}")

age_column = [col for col in df.columns if 'age' in col.lower()][0]
df_filtered = df[(df[age_column] >= 18) & (df[age_column] <= 22)].copy()

print(f"Anzahl der Datensätze nach Filterung (18-22 Jahre): {len(df_filtered)}")
display(df_filtered.head())

gender_column = [col for col in df_filtered.columns if 'gender' in col.lower()][0]

df_filtered[gender_column] = pd.to_numeric(df_filtered[gender_column], errors='coerce')
df_filtered[gender_column] = df_filtered[gender_column].map({0: 'M', 1: 'F'})

print("Verteilung des Geschlechts nach der Anpassung:")
print(df_filtered[gender_column].value_counts(dropna=False))
display(df_filtered.head())

numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns

summary_data = {}

for col in numeric_cols:
    col_data = df_filtered[col].dropna()
    n = len(col_data)

    if n == 0:
        continue

    mean_val = col_data.mean()
    median_val = col_data.median()

    mode_result = col_data.mode()
    mode_val = mode_result.iloc[0] if not mode_result.empty else np.nan

    std_val = col_data.std()
    min_val = col_data.min()
    max_val = col_data.max()
    range_val = max_val - min_val
    sum_val = col_data.sum()

    if n > 1 and std_val > 0:
        sem = stats.sem(col_data)
        confidence_level = 0.95
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

df_summary = pd.DataFrame(summary_data).T
display(df_summary)

anxiety_col = [col for col in df_filtered.columns if 'anxiety' in col.lower()][0]
headache_col = [col for col in df_filtered.columns if 'headache' in col.lower()][0]
loneliness_col = [col for col in df_filtered.columns if 'loneliness' in col.lower()][0]
gender_col = [col for col in df_filtered.columns if 'gender' in col.lower()][0]

grouped_stats = df_filtered.groupby(gender_col)[[anxiety_col, headache_col, loneliness_col]].agg(['mean', 'std']).round(3)
print("Tabellarischer Vergleich (Mittelwert & Standardabweichung):")
display(grouped_stats)

df_melted = df_filtered.melt(id_vars=[gender_col], value_vars=[anxiety_col, headache_col, loneliness_col],
                             var_name='Indikator', value_name='Ausprägung')

plt.figure(figsize=(12, 6))
sns.barplot(x='Indikator', y='Ausprägung', hue=gender_col, data=df_melted, palette='muted', errorbar='ci')
plt.title('Vergleich von Stressindikatoren zwischen weiblichen und männlichen Studierenden (95% CI)')
plt.ylabel('Durchschnittliche Ausprägung')
plt.xlabel('Stressindikator')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

stress_exp_col = [col for col in df_filtered.columns if 'stress_experience' in col.lower() or 'stress_level' in col.lower() or 'stress' in col.lower()][0]

y = df_filtered[stress_exp_col]

features = [col for col in df_filtered.select_dtypes(include=[np.number]).columns if col != stress_exp_col and col != gender_column and col != age_column]
X = df_filtered[features]

model = LinearRegression()
model.fit(X, y)

r_sq = model.score(X, y)
coefficients = pd.DataFrame({
    'Faktor': features,
    'Koeffizient (Gewichtung)': model.coef_
})
coefficients['Absoluter Einfluss'] = coefficients['Koeffizient (Gewichtung)'].abs()
coefficients = coefficients.sort_values(by='Absoluter Einfluss', ascending=False).reset_index(drop=True)

print(f"Bestimmtheitsmaß (R² - Erklärte Varianz): {r_sq:.3f} (entspricht {r_sq*100:.1f}% der Varianz)")
print("\nEinflussreichste Faktoren im Regressionsmodell:")
display(coefficients.round(4))

plt.figure(figsize=(12, 8))
sns.barplot(x='Koeffizient (Gewichtung)', y='Faktor', data=coefficients, palette='viridis')
plt.title(f'Einfluss der Faktoren auf das Stressempfinden (Erklärte Varianz R² = {r_sq:.3f})')
plt.xlabel('Regressionskoeffizient')
plt.ylabel('Faktor')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

home_env_cols = [col for col in df_filtered.columns if 'family' in col.lower() or 'home' in col.lower()]
uni_env_cols = [col for col in df_filtered.columns if 'academic' in col.lower() or 'study' in col.lower() or 'class' in col.lower()]

all_env_cols = home_env_cols + uni_env_cols

correlations = df_filtered[all_env_cols + [stress_exp_col]].corr()[[stress_exp_col]].drop(stress_exp_col)
correlations.columns = ['Korrelation mit allgemeinem Stress']
correlations = correlations.sort_values(by='Korrelation mit allgemeinem Stress', ascending=False)

print("Einfluss der Umfelder auf das Stressempfinden (Korrelationskoeffizienten):")
display(correlations.round(3))

plt.figure(figsize=(10, 6))
sns.heatmap(df_filtered[all_env_cols + [stress_exp_col]].corr()[[stress_exp_col]].sort_values(by=stress_exp_col, ascending=False),
            annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".3f")
plt.title('Korrelation von häuslichen & universitären Faktoren mit allgemeinem Stress')
plt.tight_layout()
plt.show()

features_to_cluster = df_filtered.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filtered[features_to_cluster].dropna())

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df_filtered.dropna(subset=features_to_cluster).copy()
df_clustered['Cluster'] = clusters

cluster_profile = df_clustered.groupby('Cluster')[features_to_cluster].mean().round(2)
print("Mittelwerte der Stressindikatoren je identifiziertem Studierenden-Typ (Cluster):")
display(cluster_profile)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_clustered['PCA 1'] = X_pca[:, 0]
df_clustered['PCA 2'] = X_pca[:, 1]

plt.figure(figsize=(11, 7))
sns.scatterplot(x='PCA 1', y='PCA 2', hue='Cluster', data=df_clustered, palette='Set1', alpha=0.7, style='Cluster')
plt.title('Identifizierte Studierenden-Typen (PCA 2D-Projektion der Cluster)')
plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
