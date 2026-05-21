import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import warnings

warnings.filterwarnings('ignore')

# ========== ЗАГРУЗКА ==========
df = pd.read_excel('Data_exam_st.xlsx', header=1)

print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
print(f"Колонки: {df.columns.tolist()[:10]}...")

# ========== ПОДГОТОВКА ДАННЫХ ==========
# Удаляем строки с пропусками в ключевых полях
df_clean = df.dropna(subset=['Возраст, лет', 'Онк/Неонк', 'Пол']).copy()
print(f"После удаления пропусков: {len(df_clean)} строк")

# Преобразуем пол в число
df_clean['Пол_число'] = df_clean['Пол'].map({'М': 0, 'Ж': 1, 'м': 0, 'ж': 1})

# Создаем возрастные группы
df_clean['age_group'] = pd.cut(df_clean['Возраст, лет'],
                               bins=[0, 40, 50, 60, 70, 80, 120],
                               labels=['<40', '40-49', '50-59', '60-69', '70-79', '80+'])

print("\n" + "=" * 60)
print("ЗАДАЧА 1: КОРРЕЛЯЦИЯ ВОЗРАСТА И ВЕРОЯТНОСТИ ОНКОЛОГИИ")
print("=" * 60)

# Корреляция
corr, p_val = pearsonr(df_clean['Возраст, лет'], df_clean['Онк/Неонк'])
print(f"Корреляция Пирсона: {corr:.3f} (p={p_val:.4f})")

# T-test
from scipy.stats import ttest_ind

age_onc = df_clean[df_clean['Онк/Неонк'] == 1]['Возраст, лет']
age_non = df_clean[df_clean['Онк/Неонк'] == 0]['Возраст, лет']
t_stat, t_p = ttest_ind(age_onc, age_non)
print(f"T-test: t={t_stat:.3f}, p={t_p:.4f}")
print(f"Средний возраст: Онко={age_onc.mean():.1f} лет, Не онко={age_non.mean():.1f} лет")

# График
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='Онк/Неонк', y='Возраст, лет', data=df_clean, ax=axes[0])
axes[0].set_title('Возраст: Онко vs Не онко')

age_by_group = df_clean.groupby('age_group')['Онк/Неонк'].mean() * 100
age_by_group.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Доля онкологии по возрастам')
axes[1].set_ylabel('%')
axes[1].tick_params(axis='x', rotation=45)

sns.regplot(x='Возраст, лет', y='Онк/Неонк', data=df_clean, logistic=True,
            y_jitter=0.05, scatter_kws={'alpha': 0.3}, ax=axes[2])
axes[2].set_title('Логистическая регрессия')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ЗАДАЧА 2: ПОИСК ПОРОГОВОГО ВОЗРАСТА")
print("=" * 60)

thresholds = range(40, 75, 1)
or_values = []
for thr in thresholds:
    high_age = (df_clean['Возраст, лет'] >= thr).astype(int)
    ct = pd.crosstab(high_age, df_clean['Онк/Неонк'])
    if ct.shape == (2, 2) and ct.min().min() > 0:
        or_val = (ct.iloc[0, 0] * ct.iloc[1, 1]) / (ct.iloc[0, 1] * ct.iloc[1, 0])
        or_values.append((thr, or_val))

thr_vals, or_vals = zip(*or_values)
plt.figure(figsize=(10, 5))
plt.plot(thr_vals, or_vals, 'o-', color='darkblue')
plt.axhline(y=2, color='red', linestyle='--', label='OR=2')
plt.xlabel('Пороговый возраст (≥ X лет)')
plt.ylabel('Odds Ratio')
plt.title('Риск онкологии в зависимости от порога возраста')
plt.grid(True, alpha=0.3)
plt.legend()

# Находим критический возраст
for thr, or_val in or_values:
    if or_val > 2:
        plt.axvline(x=thr, color='green', linestyle='--', alpha=0.7)
        plt.text(thr, max(or_vals) * 0.8, f'{thr} лет', rotation=90, color='green')
        print(f"⚠️ КРИТИЧЕСКИЙ ПОРОГ: {thr} лет (OR={or_val:.2f})")
        break
plt.show()

print("\n" + "=" * 60)
print("ЗАДАЧА 3: КОМОРБИДНЫЕ СОСТОЯНИЯ")
print("=" * 60)

# Ищем колонки с коморбидностями
comorbid_cols = ['Коморбидные состояния', 'Коморбидная патология', 'Курение']
existing_comorbid = [c for c in comorbid_cols if c in df_clean.columns]

if existing_comorbid:
    for col in existing_comorbid:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].map({'да': 1, 'нет': 0})

    # Частота по группам
    results = []
    for col in existing_comorbid:
        for onc_val in [0, 1]:
            subset = df_clean[df_clean['Онк/Неонк'] == onc_val]
            freq = subset[col].mean() * 100
            results.append({'Коморбидность': col, 'Группа': 'Онко' if onc_val == 1 else 'Не онко', 'Частота, %': freq})

    res_df = pd.DataFrame(results)
    pivot = res_df.pivot(index='Коморбидность', columns='Группа', values='Частота, %')

    # График
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pivot.plot(kind='barh', ax=axes[0])
    axes[0].set_title('Частота коморбидностей')

    # Статзначимость
    sig_data = []
    for col in existing_comorbid:
        ct = pd.crosstab(df_clean[col], df_clean['Онк/Неонк'])
        if ct.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(ct)
            sig_data.append({'Коморбидность': col, 'p-value': p})

    if sig_data:
        sig_df = pd.DataFrame(sig_data)
        colors = ['green' if p < 0.05 else 'red' for p in sig_df['p-value']]
        axes[1].barh(sig_df['Коморбидность'], -np.log10(sig_df['p-value'] + 1e-10), color=colors)
        axes[1].axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1].set_xlabel('-log10(p-value)')
        axes[1].set_title('Статистическая значимость')
        axes[1].legend()

        print("\nРезультаты:")
        for _, row in sig_df.iterrows():
            status = "✅ значимо" if row['p-value'] < 0.05 else "❌ не значимо"
            print(f"  {row['Коморбидность']}: p={row['p-value']:.4f} - {status}")

    plt.tight_layout()
    plt.show()
else:
    print("Колонки с коморбидностями не найдены")

print("\n" + "=" * 60)
print("ЗАДАЧА 4: МОДЕЛЬ ПРОГНОЗА")
print("=" * 60)

# Подготовка признаков
features = ['Возраст, лет', 'Пол_число'] + existing_comorbid
X = df_clean[features].fillna(0)
y = df_clean['Онк/Неонк']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Не онко', 'Онко']))

print("\nВАЖНОСТЬ ПРИЗНАКОВ:")
for feat, coef in zip(features, model.coef_[0]):
    direction = "↑ риск" if coef > 0 else "↓ риск"
    print(f"  {feat}: {coef:.3f} ({direction})")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_score(y_test, y_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "=" * 60)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 60)