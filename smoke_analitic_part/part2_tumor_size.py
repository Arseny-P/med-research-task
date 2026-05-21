import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display

def analyze_tumor_size(df):
    smokers = df[(df['Курение'] == 'да') & (df['Сред. размер, мм'].notna())]['Сред. размер, мм']
    non_smokers = df[(df['Курение'] == 'нет') & (df['Сред. размер, мм'].notna())]['Сред. размер, мм']
    
    stats_dict = {
        'Группа': ['Курильщики', 'Некурящие'],
        'N': [len(smokers), len(non_smokers)],
        'Среднее (мм)': [f"{smokers.mean():.2f}", f"{non_smokers.mean():.2f}"],
        'Медиана (мм)': [f"{smokers.median():.2f}", f"{non_smokers.median():.2f}"],
        'Стд. отклонение': [f"{smokers.std():.2f}", f"{non_smokers.std():.2f}"],
        'Min (мм)': [f"{smokers.min():.2f}", f"{non_smokers.min():.2f}"],
        'Max (мм)': [f"{smokers.max():.2f}", f"{non_smokers.max():.2f}"]
    }
    
    stats_df = pd.DataFrame(stats_dict)
    
    print("\nОПИСАТЕЛЬНАЯ СТАТИСТИКА:")
    display(stats_df.style.set_caption("Характеристики размера очага"))
    
    diff = smokers.mean() - non_smokers.mean()
    diff_percent = (diff / non_smokers.mean()) * 100
    
    print(f"\nРазница средних: {diff:.2f} мм ({diff_percent:+.1f}%)")

    print("\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ ГРУПП:")
    

    t_stat, p_value = stats.ttest_ind(smokers, non_smokers, equal_var=False)
    test_name = "T-тест Стьюдента (непарный)"

    test_results = pd.DataFrame({
        'Тест': [test_name],
        'Статистика': [f"{t_stat:.4f}"],
        'p-value': [f"{p_value:.6f}"],
        'Значимость': ['Значимо (p < 0.05)' if p_value < 0.05 else 'Не значимо (p ≥ 0.05)']
    })

    display(test_results.style.set_caption("Статистический тест"))

    t_stat, p_value = stats.mannwhitneyu(smokers, non_smokers)
    test_name = "U-тест Манна-Уитни"
    
    test_results = pd.DataFrame({
        'Тест': [test_name],
        'Статистика': [f"{t_stat:.4f}"],
        'p-value': [f"{p_value:.6f}"],
        'Значимость': ['Значимо (p < 0.05)' if p_value < 0.01 else 'Не значимо (p ≥ 0.05)']
    })

    display(test_results.style.set_caption("Статистический тест"))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot
    bp_data = [smokers.values, non_smokers.values]
    bp = axes[0].boxplot(bp_data, patch_artist=True, 
                         labels=['Курильщики', 'Некурящие'])
    
    colors = ['#FF9999', '#99CCFF']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    axes[0].set_title("Распределение размеров (Boxplot)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Размер очага, мм")
    axes[0].grid(True, alpha=0.3)
    
    # Гистограмма
    axes[1].hist(smokers, alpha=0.6, label='Курильщики', color='#FF9999', 
                bins=20, edgecolor='black')
    axes[1].hist(non_smokers, alpha=0.6, label='Некурящие', color='#99CCFF', 
                bins=20, edgecolor='black')
    axes[1].axvline(smokers.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Среднее курящие: {smokers.mean():.1f}')
    axes[1].axvline(non_smokers.mean(), color='darkblue', linestyle='--', linewidth=2, 
                   label=f'Среднее некурящие: {non_smokers.mean():.1f}')
    axes[1].set_title("Распределение (гистограмма)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Размер очага, мм")
    axes[1].set_ylabel("Частота")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("ЧАСТЬ 2: Сравнение размера очага", fontsize=14, fontweight='bold')
    plt.tight_layout()
    # plt.savefig('part2.png')
    plt.show()