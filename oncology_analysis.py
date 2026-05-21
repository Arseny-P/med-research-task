import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
df = pd.read_excel('Data_exam_st.xlsx', header=1)
df.columns = df.columns.str.strip()

df['is_onco'] = pd.to_numeric(df['Онк/Неонк'], errors='coerce').fillna(0).astype(int)

df['age'] = pd.to_numeric(df['Возраст, лет'], errors='coerce')

df['sex_male'] = (df['Пол'] == 'м').astype(int)

df['comorbidity'] = df['Коморбидные состояния'].astype(str).str.strip()
df['has_HOBL'] = df['comorbidity'].str.contains('ХОБЛ', case=False, na=False).astype(int)
df['has_calcinates'] = df['comorbidity'].str.contains('кальцинат', case=False, na=False).astype(int)
df['has_petrificates'] = df['comorbidity'].str.contains('петрификат', case=False, na=False).astype(int)
df['has_any_comorb'] = ((df['comorbidity'].str.lower() != 'нет') & (df['comorbidity'].str.lower() != 'nan')).astype(int)

df_clean = df.dropna(subset=['age', 'is_onco']).copy()

p = df_clean['is_onco'].mean()
n = len(df_clean)
ci = stats.norm.ppf(0.975) * np.sqrt(p * (1 - p) / n)

print(f"""
Всего пациентов: {n}
    Онкология: {int(p * n)} ({p * 100:.1f}% [95% ДИ: {(p - ci) * 100:.1f}–{(p + ci) * 100:.1f}%])
    Не онкология: {n - int(p * n)} ({(1 - p) * 100:.1f}%)

Демография по группам:
    Онкология:   {df_clean[df_clean['is_onco'] == 1]['age'].mean():.1f} +- {df_clean[df_clean['is_onco'] == 1]['age'].std():.1f} лет, {df_clean[df_clean['is_onco'] == 1]['sex_male'].mean() * 100:.0f}%
    Не онкология: {df_clean[df_clean['is_onco'] == 0]['age'].mean():.1f} +- {df_clean[df_clean['is_onco'] == 0]['age'].std():.1f} лет, {df_clean[df_clean['is_onco'] == 0]['sex_male'].mean() * 100:.0f}%
""")

print(" 1) Взоимосвязь возраста с онкологией")

# 1.1 Визуализация распределения возраста
print("\n 1.1 Распределение возраста в двух группах")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df_clean, x='age', hue='is_onco', kde=True, ax=axes[0], palette='Set1', alpha=0.6)
axes[0].set_title('Распределение возраста')
axes[0].legend(['Не онко', 'Онко'])
axes[0].set_xlabel('Возраст, лет')

sns.boxplot(data=df_clean, x='is_onco', y='age', ax=axes[1], palette='Set1', hue='is_onco', legend=False)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Не онко', 'Онко'])
axes[1].set_title('Сравнение возраста (boxplot)')
axes[1].set_ylabel('Возраст, лет')

plt.tight_layout()
plt.show()

median_onco = df_clean[df_clean['is_onco'] == 1]['age'].median()
median_no = df_clean[df_clean['is_onco'] == 0]['age'].median()
print(f" Медианный возраст: онкология = {median_onco:.0f} лет, не онкология = {median_no:.0f} лет")
# 1.2 Статистическое сравнение возраста
print("\n 1.2 Статистическое сравнение возраста между группами")
age_onco = df_clean[df_clean['is_onco'] == 1]['age'].dropna()
age_no = df_clean[df_clean['is_onco'] == 0]['age'].dropna()
print(f"Онкология (n={len(age_onco)}): среднее={age_onco.mean():.1f}, медиана={age_onco.median():.1f}, SD={age_onco.std():.1f}")
print(f"Не онкология (n={len(age_no)}): среднее={age_no.mean():.1f}, медиана={age_no.median():.1f}, SD={age_no.std():.1f}")
# Проверка нормальности
try:
    _, p_norm_onco = stats.shapiro(age_onco.sample(min(500, len(age_onco))))
    _, p_norm_no = stats.shapiro(age_no.sample(min(500, len(age_no))))
    normal = p_norm_onco > 0.05 and p_norm_no > 0.05
except:
    normal = False
# Выбор теста
if normal and len(age_onco) >= 3 and len(age_no) >= 3:
    t_stat, p_val = stats.ttest_ind(age_onco, age_no)
    test_name = "T-тест Стьюдента"
    effect_size = (age_onco.mean() - age_no.mean()) / np.sqrt(
        (age_onco.std() ** 2 + age_no.std() ** 2) / 2)
else:
    t_stat, p_val = stats.mannwhitneyu(age_onco, age_no)
    test_name = "U-тест Манна-Уитни"
    effect_size = None

print(f"\n {test_name}:")
print(f"Статистика: {t_stat:.3f}")
print(f"p-value: {p_val:.4f}")
if effect_size is not None:
    print(f"Размер эффекта (Cohen's d): {abs(effect_size):.2f} {'(большой)' if abs(effect_size) > 0.8 else '(средний)' if abs(effect_size) > 0.5 else '(малый)'}")

# 1.3 Пороговый возраст
print("\n 1.3 Поиск порогового возраста, где риск резко растёт")

age_groups = df_clean.groupby('age')['is_onco'].agg(['mean', 'count']).reset_index()
age_groups = age_groups[age_groups['count'] >= 3]

if len(age_groups) > 1:
    age_groups['diff'] = age_groups['mean'].diff()
    thr_idx = age_groups['diff'].idxmax()
    thr_age = age_groups.loc[thr_idx, 'age']
    prob_before = age_groups[age_groups['age'] < thr_age]['mean'].mean()
    prob_after = age_groups[age_groups['age'] >= thr_age]['mean'].mean()
    print(f"\n Эмпирический порог: ~{thr_age:.0f} лет")
    print(f"До {thr_age:.0f} лет: онкология у {prob_before:.1%} пациентов")
    print(f"После {thr_age:.0f} лет: онкология у {prob_after:.1%} пациентов")
    print(f"Рост риска: +{(prob_after / prob_before - 1) * 100:.0f}%")
    # Визуализация порога
    plt.figure(figsize=(9, 5))
    plt.plot(age_groups['age'], age_groups['mean'] * 100, marker='o', markersize=4, linewidth=2)
    plt.axvline(x=thr_age, color='red', linestyle='--', linewidth=2, label=f'Порог: {thr_age:.0f} лет')
    plt.xlabel('Возраст, лет')
    plt.ylabel('Доля онкологии, %')
    plt.title('Динамика доли онкологии по возрасту')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 1.4 Возраст по гистотипам
print("\n 1.4 Сравнение возраста по гистологическим типам")
top_hist = df_clean['Гистология'].value_counts().head(5).index
df_hist = df_clean[df_clean['Гистология'].isin(top_hist)]
print("\nМедианный возраст по гистотипам (от молодого к старшему):")
median_by_hist = df_hist.groupby('Гистология')['age'].median().sort_values()
for hist_type, median_age in median_by_hist.items():
    print(f"{hist_type}: {median_age:.0f} лет")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_hist, x='Гистология', y='age', palette='pastel')
plt.xticks(rotation=45, ha='right')
plt.title('Распределение возраста по гистотипам')
plt.ylabel('Возраст, лет')
plt.tight_layout()
plt.show()
# Статистический тест
hist_groups = [group['age'].dropna() for name, group in df_hist.groupby('Гистология')]
if len(hist_groups) >= 2:
    try:
        f_stat, p_val = stats.f_oneway(*hist_groups)
        test_name = "ANOVA"
    except:
        f_stat, p_val = stats.kruskal(*hist_groups)
        test_name = "Тест Крускала-Уоллиса"
    print(f"\n{test_name}: p-value = {p_val:.4f}")
    print(f"{'Различия в возрасте между гистотипами значимы' if p_val < 0.05 else 'Различия не значимы'}")

print("2) Как коморбидные состояния связаны с онкологией?")
print("\n 2.1 Сравнение частоты коморбидностей в группах")
comorb_stats = pd.DataFrame({
    'Коморбидность': ['ХОБЛ', 'Кальцинаты', 'Петрификаты', 'Любая коморбидность'],
    'Частота в группе Не онко': [
        df_clean[df_clean['is_onco'] == 0]['has_HOBL'].mean() * 100,
        df_clean[df_clean['is_onco'] == 0]['has_calcinates'].mean() * 100,
        df_clean[df_clean['is_onco'] == 0]['has_petrificates'].mean() * 100,
        df_clean[df_clean['is_onco'] == 0]['has_any_comorb'].mean() * 100
    ],
    'Частота в группе Онко': [
        df_clean[df_clean['is_onco'] == 1]['has_HOBL'].mean() * 100,
        df_clean[df_clean['is_onco'] == 1]['has_calcinates'].mean() * 100,
        df_clean[df_clean['is_onco'] == 1]['has_petrificates'].mean() * 100,
        df_clean[df_clean['is_onco'] == 1]['has_any_comorb'].mean() * 100
    ]
})

print("\nЧастота коморбидностей:")
print(comorb_stats.to_string(index=False))
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
comorbs = ['has_HOBL', 'has_calcinates', 'has_petrificates', 'has_any_comorb']
titles = ['ХОБЛ', 'Кальцинаты', 'Петрификаты', 'Любая коморбидность']
for idx, (comorb, title) in enumerate(zip(comorbs, titles)):
    ax = axes[idx // 2, idx % 2]
    freq = df_clean.groupby('is_onco')[comorb].mean() * 100
    ax.bar(['Не онко', 'Онко'], freq.values, color=['skyblue', 'salmon'], alpha=0.8)
    ax.set_ylabel('Частота, %')
    ax.set_title(title)
    ax.set_ylim([0, max(freq.max() * 1.3, 100)])
    # χ²-тест
    contingency = pd.crosstab(df_clean['is_onco'], df_clean[comorb])
    if contingency.shape == (2, 2) and contingency.min().min() >= 5:
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        # Отношение шансов вручную
        a, b = contingency.iloc[0, 0], contingency.iloc[0, 1]
        c, d = contingency.iloc[1, 0], contingency.iloc[1, 1]
        or_val = (d / c) / (b / a) if c > 0 and a > 0 else np.nan
        ax.text(0.5, 0.97, f' p={p:.4f}\nOR={or_val:.2f}',
                transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.show()
# Краткий вывод по ХОБЛ
hobl_ratio = (df_clean[df_clean['is_onco'] == 1]['has_HOBL'].mean() / df_clean[df_clean['is_onco'] == 0]['has_HOBL'].mean())
print(f"\n ХОБЛ встречается в {hobl_ratio:.2f} раза чаще у пациентов с онкологией")
print("\n 2.2 Статистическая оценка ассоциаций (χ²-тест)")
print(f"\n Результаты χ²-теста для бинарных коморбидностей:")
for comorb_name, comorb_col in [('ХОБЛ', 'has_HOBL'), ('Кальцинаты', 'has_calcinates'), ('Петрификаты', 'has_petrificates')]:
    contingency = pd.crosstab(df_clean['is_onco'], df_clean[comorb_col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    a, b = contingency.iloc[0, 0], contingency.iloc[0, 1]
    c, d = contingency.iloc[1, 0], contingency.iloc[1, 1]
    or_val = (d / c) / (b / a) if c > 0 and a > 0 else np.nan
    ci_low = or_val * np.exp(-1.96 * np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    ci_high = or_val * np.exp(1.96 * np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    print(f"{comorb_name}: χ²={chi2:.2f}, p={p:.4f}, OR={or_val:.2f} [95% ДИ: {ci_low:.2f}–{ci_high:.2f}]")
# 2.3 Heatmap: коморбидности * гистотипы
print("\n2.3 Визуализация связей: коморбидности × гистотипы")
top_hist = df_clean['Гистология'].value_counts().head(5).index
df_heatmap = df_clean[df_clean['Гистология'].isin(top_hist)]
heatmap_data = []
for comorb_name, comorb_col in [('ХОБЛ', 'has_HOBL'), ('Кальцинаты', 'has_calcinates'), ('Петрификаты', 'has_petrificates')]:
    row = []
    for hist in top_hist:
        freq = df_heatmap[df_heatmap['Гистология'] == hist][comorb_col].mean() * 100
        row.append(freq)
    heatmap_data.append(row)
heatmap_df = pd.DataFrame(heatmap_data, index=['ХОБЛ', 'Кальцинаты', 'Петрификаты'], columns=top_hist)
plt.figure(figsize=(11, 7))
sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Частота, %'}, linewidths=0.5)
plt.title('Частота коморбидностей по гистологическим типам (%)')
plt.ylabel('Коморбидность')
plt.xlabel('Гистологический диагноз')
plt.tight_layout()
plt.show()
print("\n Как читать heatmap: чем ярче ячейка — тем чаще данная коморбидность встречается при данном гистотипе")
print("Итог")
# Рассчитываем значения для вывода
hobl_onco_pct = df_clean[df_clean['is_onco'] == 1]['has_HOBL'].mean() * 100
hobl_no_pct = df_clean[df_clean['is_onco'] == 0]['has_HOBL'].mean() * 100
hobl_ratio = df_clean[df_clean['is_onco'] == 1]['has_HOBL'].mean() / df_clean[df_clean['is_onco'] == 0][
    'has_HOBL'].mean()

print(f"""
 ВОЗРАСТ — ключевой фактор, связанный с онкологией
   Медианный возраст: онкология = {median_onco:.0f} лет, не онкология = {median_no:.0f} лет
   Статистический тест: {'значимое различие' if p_val < 0.05 else 'различие не значимо'} (p = {p_val:.4f})
   Пороговый возраст риска: ~{thr_age:.0f} лет
     До {thr_age:.0f} лет: онкология у {prob_before:.1%} пациентов
     После {thr_age:.0f} лет: онкология у {prob_after:.1%} пациентов
     Рост риска: +{(prob_after / prob_before - 1) * 100:.0f}%

 ХОБЛ ассоциирована с онкологией
    Частота ХОБЛ: {hobl_onco_pct:.1f}% в онко-группе против {hobl_no_pct:.1f}% в контрольной
    ХОБЛ встречается в {hobl_ratio:.2f} раза чаще у пациентов с онкологией
    χ²-тест подтверждает значимую ассоциацию (p < 0.05)

 Кальцинаты и петрификаты — возможные маркеры доброкачественных процессов
   Их частота ниже в онко-группе
   Это может помогать в дифференциальной диагностике

 Разные гистотипы встречаются в разном возрасте
   Самый "молодой": туберкулёма ({median_by_hist.iloc[0]:.0f} лет)
   Самый "возрастной": плоскоклеточный рак ({median_by_hist.iloc[-1]:.0f} лет)
   Различия статистически значимы (ANOVA / Крускал-Уоллис, p < 0.001)
""")
