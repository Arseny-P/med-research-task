import matplotlib.pyplot as plt
from metrics import get_calculated_data

data = get_calculated_data()

plt.figure(figsize=(7, 5))
plt.plot(data['fpr_18f'], data['tpr_18f'], color='darkorange', linewidth=3, label=f"18F-ФДГ (Точность: {data['auc_18f']:.2f})")
plt.plot(data['fpr_11c'], data['tpr_11c'], color='green', linewidth=3, label=f"11C-метионин (Точность: {data['auc_11c']:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Специфичность', fontsize=10)
plt.ylabel('Чувствительность', fontsize=10)
plt.title('Сравнение диагностической эффективности препаратов', fontsize=11, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()

plt.show();
# plt.savefig('roc_curves.png')
plt.close()


plt.figure(figsize=(6, 5))
plt.boxplot([data['onco_18f'], data['nononco_18f']], labels=['Онкология', 'Не онкология'], patch_artist=True,
            boxprops=dict(facecolor='bisque', color='darkorange', linewidth=1.5))
plt.axhline(y=data['thresh_18f'], color='red', linestyle='--', linewidth=2, label=f"Порог разделения ({data['thresh_18f']:.2f})")
plt.ylabel('Значение SUVочаг', fontsize=10)
plt.title('Разброс и порог значений SUV для 18F-ФДГ', fontsize=11, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.2, axis='y')
plt.tight_layout()

plt.show();
# plt.savefig('18f.png')
plt.close()


plt.figure(figsize=(6, 5))
plt.boxplot([data['onco_11c'], data['nononco_11c']], labels=['Онкология', 'Не онкология'], patch_artist=True,
            boxprops=dict(facecolor='lightgreen', color='green', linewidth=1.5))
plt.axhline(y=data['thresh_11c'], color='red', linestyle='--', linewidth=2, label=f"Порог разделения ({data['thresh_11c']:.2f})")
plt.ylabel('Значение SUVочаг', fontsize=10)
plt.title('Разброс и порог значений SUV для 11C-метионина', fontsize=11, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.2, axis='y')
plt.tight_layout()

plt.show();
# plt.savefig('11c.png')