import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

def analyze_interaction(df):
    bins = [0, 40, 50, 60, 70, 100]
    labels = ['<40', '40-50', '50-60', '60-70', '>70']
    df['age_group'] = pd.cut(df['Возраст, лет'], bins=bins, labels=labels)
    
    pivot_table = df.groupby(['Курение', 'age_group']).agg({
        'Сред. размер, мм': ['mean', 'std', 'count'],
        'Онк/Неонк': 'mean'
    }).round(2)
    
    print("\nСРЕДНИЙ РАЗМЕР ОЧАГА ПО ВОЗРАСТНЫМ ГРУППАМ:")
    display(pivot_table.style.set_properties(**{'text-align': 'center'})
            .set_caption("Статистика по возрастным группам и курению"))
    
    # Корреляции
    corr_smokers = df[df['Курение'] == 'да'][['Возраст, лет', 'Сред. размер, мм']].corr()
    corr_non_smokers = df[df['Курение'] == 'нет'][['Возраст, лет', 'Сред. размер, мм']].corr()
    
    corr_df = pd.DataFrame({
        'Группа': ['Курильщики', 'Некурящие'],
        'Корреляция (возраст-размер)': [
            f"r = {corr_smokers.values[0,1]:.3f}",
            f"r = {corr_non_smokers.values[0,1]:.3f}"
        ]
    })
    

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Тепловая карта
    pivot_heatmap = df.pivot_table(values='Сред. размер, мм', 
                                   index='Курение', 
                                   columns='age_group', 
                                   aggfunc='mean')
    im = axes[0].imshow(pivot_heatmap.values, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(len(pivot_heatmap.columns)))
    axes[0].set_yticks(range(len(pivot_heatmap.index)))
    axes[0].set_xticklabels(pivot_heatmap.columns)
    axes[0].set_yticklabels(['Курильщики', 'Некурящие'])
    axes[0].set_xlabel('Возрастная группа', fontsize=11)
    axes[0].set_ylabel('Курение', fontsize=11)
    axes[0].set_title('Тепловая карта среднего размера', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Размер, мм')
    
    for i in range(len(pivot_heatmap.index)):
        for j in range(len(pivot_heatmap.columns)):
            value = pivot_heatmap.values[i, j]
            if not np.isnan(value):
                axes[0].text(j, i, f'{value:.1f}', 
                               ha='center', va='center', 
                               color='black', fontweight='bold')
    
    # Доля онкологии
    pivot_onco = df.groupby(['Курение', 'age_group'])['Онк/Неонк'].mean().unstack() * 100
    pivot_onco.plot(kind='bar', ax=axes[1], 
                   edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Возрастная группа', fontsize=11)
    axes[1].set_ylabel('Доля онкологии, %', fontsize=11)
    axes[1].set_title('Частота онкологии по возрастам', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("ЧАСТЬ 4: Многомерный анализ", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    # plt.savefig('part4.png')
    plt.show()