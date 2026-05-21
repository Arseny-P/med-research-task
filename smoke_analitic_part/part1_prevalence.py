import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from IPython.display import display

def analyze_prevalence(df):
    table = pd.crosstab(df['Курение'], df['Онк/Неонк'], 
                    rownames=['Курение'], 
                    colnames=['Онкология'])
    
    print("\nТАБЛИЦА СОПРЯЖЕННОСТИ:")
    display(table.style.background_gradient(cmap='Blues', axis=None)
            .set_caption("Распределение пациентов по группам"))
    
    chi2, p_value, dof, expected = chi2_contingency(table)
    
    results_df = pd.DataFrame({
        'Параметр': ['χ² (хи-квадрат)', 'p-value', 'Степени свободы'],
        'Значение': [f"{chi2:.4f}", f"{p_value:.6f}", dof]
    })
    
    print("\nРЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО ТЕСТА:")
    display(results_df.style.set_caption("Хи-квадрат тест"))
    
    if p_value < 0.05:
        print("\nВЫВОД: Связь между курением и онкологией СТАТИСТИЧЕСКИ ЗНАЧИМА (p < 0.05)")
    else:
        print("\nВЫВОД: Связь НЕ является статистически значимой (p ≥ 0.05)")
    

    percent_table = pd.crosstab(df['Курение'], df['Онк/Неонк'], 
                                 normalize='index') * 100
    percent_table.columns = ['Без онкологии (%)', 'Онкология (%)']
    
    print("\nПРОЦЕНТНОЕ РАСПРЕДЕЛЕНИЕ:")
    display(percent_table.style.format("{:.1f}%")
            .set_caption("Доля пациентов в процентах"))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Столбчатая диаграмма
    table.plot(kind='bar', ax=axes[0], color=['#FF9999', '#66B2FF'], edgecolor='black')
    axes[0].set_title("Абсолютное распределение", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Количество пациентов")
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(['Без онкологии', 'Онкология'])
    
    # Круговая диаграмма для группы с онкологией
    onco_smoking = df[df['Онк/Неонк'] == 1]['Курение'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    axes[1].pie(onco_smoking, labels=['Курильщики', 'Некурящие'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title("Группа ОНКОЛОГИЯ", fontsize=12, fontweight='bold')
    
    plt.suptitle("ЧАСТЬ 1: Распространенность курения", fontsize=14, fontweight='bold', y=1)
    plt.tight_layout()
    # plt.savefig('part1.png')
    plt.show()