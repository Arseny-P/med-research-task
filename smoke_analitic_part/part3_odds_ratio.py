import pandas as pd
import matplotlib.pyplot as plt
import math
from IPython.display import display

def analyze_odds_ratio(df):
    # === Создаем таблицу 2×2 ===
    a = len(df[(df['Курение'] == 'да') & (df['Онк/Неонк'] == 1)])
    b = len(df[(df['Курение'] == 'да') & (df['Онк/Неонк'] == 0)])
    c = len(df[(df['Курение'] == 'нет') & (df['Онк/Неонк'] == 1)])
    d = len(df[(df['Курение'] == 'нет') & (df['Онк/Неонк'] == 0)])

    table_2x2 = pd.DataFrame([
        [a, b, a+b],
        [c, d, c+d],
        [a+c, b+d, a+b+c+d]
    ], columns=['Онкология (1)', 'Без онкологии (0)', 'Всего'],
       index=['Курильщики', 'Некурящие', 'Всего'])
    
    print("\n📊 ТАБЛИЦА 2×2:")
    try:
        display(table_2x2.style
                .background_gradient(cmap='Greens', axis=None, vmin=0)
                .set_caption("Распределение для расчета OR"))
    except AttributeError:
        print(table_2x2.to_string())
    

    if b == 0 or c == 0:
        print("\nПредупреждение: Невозможно рассчитать OR (деление на ноль)")
        print("В одной из ячеек таблицы 2×2 нет данных")
        return
    
    # === Расчет OR ===
    or_value = (a * d) / (b * c)
    
    # 95% Доверительный интервал
    se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
    log_or = math.log(or_value)
    ci_lower = math.exp(log_or - 1.96 * se_log_or)
    ci_upper = math.exp(log_or + 1.96 * se_log_or)

    # === Таблица результатов ===
    results_df = pd.DataFrame({
        'Показатель': ['Odds Ratio (OR)', '95% ДИ (нижняя граница)', 
                      '95% ДИ (верхняя граница)', 'ln(OR)', 'Стат. значимость'],
        'Значение': [
            f"{or_value:.4f}", 
            f"{ci_lower:.4f}", 
            f"{ci_upper:.4f}", 
            f"{log_or:.4f}",
            "Значимо" if (ci_lower > 1 or ci_upper < 1) else "Не значимо"
        ]
    })
    
    print("\nРЕЗУЛЬТАТЫ РАСЧЕТА ОТНОШЕНИЯ ШАНСОВ:")
    try:
        display(results_df.style.set_caption("Odds Ratio и доверительный интервал"))
    except:
        print(results_df.to_string(index=False))
    
    # === Интерпретация ===
    print("\nИНТЕРПРЕТАЦИЯ:")
    if or_value > 1:
        print(f"   Курение УВЕЛИЧИВАЕТ шансы развития онкологии в {or_value:.2f} раз")
        if ci_lower > 1:
            print("   Эффект СТАТИСТИЧЕСКИ ЗНАЧИМ (95% ДИ не включает 1)")
        else:
            print("   Эффект НЕ значим (95% ДИ включает 1)")
    elif or_value < 1:
        print(f"   Курение СНИЖАЕТ шансы в {1/or_value:.2f} раз")
        if ci_upper < 1:
            print("   Эффект СТАТИСТИЧЕСКИ ЗНАЧИМ")
        else:
            print("   Эффект НЕ значим")
    else:
        print("   Курение не влияет на риск")
    
    # === График Forest Plot ===
    fig, ax = plt.subplots(figsize=(8, 4))
    
    y_pos = [0]
    ax.errorbar(x=[or_value], y=y_pos, 
               xerr=[[or_value - ci_lower], [ci_upper - or_value]], 
               fmt='s', color='red', capsize=10, markersize=12, label='OR')
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=2, label='OR = 1 (нет эффекта)')
    ax.set_xlim(0, max(ci_upper * 1.3, 3))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(['Курение'])
    ax.set_xlabel('Odds Ratio', fontsize=11)
    ax.set_title(f'Forest Plot: OR = {or_value:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем текст с результатом
    ax.text(or_value, 0.15, f'OR = {or_value:.2f}\n95% ДИ: [{ci_lower:.2f}, {ci_upper:.2f}]', 
           ha='center', va='bottom', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    # plt.savefig('part3_forest_plot.png', dpi=300)
    plt.show()