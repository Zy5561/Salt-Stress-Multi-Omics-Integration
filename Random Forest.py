import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

import numpy as np
try:
    np.bool
except AttributeError:
    np.bool = bool

import pandas as pd
from sklearn.ensemble import RandomForestClassifier   # 替换 SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ==============================================================================
# 1. 全局绘图风格
# ==============================================================================
FONT_SIZE_TITLE = 26
FONT_SIZE_LABEL = 22
FONT_SIZE_TICK = 20

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = FONT_SIZE_LABEL
plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK

# ==============================================================================
# 2. 数据读取与预处理
# ==============================================================================
metabolite_path = r""
label_path = r""
output_dir = r""

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_met = pd.read_csv(metabolite_path, index_col=0, encoding='utf-8-sig')
df_label = pd.read_excel(label_path, index_col=0)

df_merged = df_met.join(df_label.iloc[:, 0], how='inner')
y_labels = df_merged.iloc[:, -1].astype(int)
X = df_merged.iloc[:, :-1].fillna(0)

print(f"✅ 数据读取完成。样本量: {len(X)}, 特征数: {X.shape[1]}, 标签类别: {set(y_labels)}")

# ==============================================================================
# 3. 嵌套交叉验证建模（随机森林）
# ==============================================================================
# 随机森林参数网格（可自行调整范围）
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
n_repeats = 100

prediction_storage = {idx: [] for idx in X.index}
all_feature_importances = []          # 记录每次折的特征重要性
all_best_params = []                  # 记录每次折的最优参数组合

print("🔄 正在执行嵌套交叉验证（100次重复，随机森林模型）...")

for r in range(n_repeats):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r * 123)

    for train_idx, test_idx in outer_cv.split(X, y_labels):
        X_train_raw = X.iloc[train_idx]
        X_test_raw  = X.iloc[test_idx]
        y_train = y_labels.iloc[train_idx]
        y_test  = y_labels.iloc[test_idx]

        # 标准化（对树模型非必需，但保留原流程；不标准化也不影响）
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            index=X_train_raw.index, columns=X_train_raw.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_raw),
            index=X_test_raw.index, columns=X_test_raw.columns
        )

        grid_search = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=r * 7 + 3),
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=r * 7 + 3),
            scoring='balanced_accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        all_best_params.append(grid_search.best_params_)

        preds = model.predict(X_test)
        for i, idx in enumerate(X_test.index):
            prediction_storage[idx].append(preds[i])

        # 记录特征重要性（随机森林原生）
        all_feature_importances.append(model.feature_importances_)

    if (r + 1) % 20 == 0:
        print(f"  ✔ 已完成 {r + 1}/{n_repeats}")

# 多数投票确定最终预测
final_preds_list = []
for idx in X.index:
    votes = prediction_storage[idx]
    most_common_label = Counter(votes).most_common(1)[0][0]
    final_preds_list.append(most_common_label)

# 统计最优参数组合的众数（用于最终全量模型）
from collections import defaultdict
param_counter = defaultdict(int)
for params in all_best_params:
    # 将参数组合转为可哈希的元组
    key = (params['n_estimators'], params['max_depth'], params['min_samples_split'])
    param_counter[key] += 1
best_params_tuple = max(param_counter.items(), key=lambda x: x[1])[0]
best_n_estimators, best_max_depth, best_min_samples_split = best_params_tuple
print(f"\n📊 100×5 折 CV 中最优参数众数: n_estimators={best_n_estimators}, max_depth={best_max_depth}, min_samples_split={best_min_samples_split}")
print(f"   参数分布: {dict(param_counter)}")

# ==============================================================================
# 4. 导出品种预测明细表
# ==============================================================================
print("\n📝 正在生成品种预测明细表...")

result_rows = []
for idx in X.index:
    votes = prediction_storage[idx]
    total = len(votes)
    count_1 = votes.count(1)
    count_neg1 = votes.count(-1)
    ratio_1 = count_1 / total
    ratio_neg1 = count_neg1 / total
    final_pred = Counter(votes).most_common(1)[0][0]

    result_rows.append({
        'Variety_ID': idx,
        'Actual_Label': y_labels.loc[idx],
        'Predicted_Label': final_pred,
        'Is_Correct': int(y_labels.loc[idx] == final_pred),
        'Ratio_Pred_1': ratio_1,
        'Ratio_Pred_-1': ratio_neg1
    })

df_results = pd.DataFrame(result_rows)
excel_path = os.path.join(output_dir, "Variety_True_vs_Predicted.xlsx")
df_results.to_excel(excel_path, index=False)
print(f"✅ 对照表已导出至: {excel_path}")

# ==============================================================================
# 5. 绘图与模型评估
# ==============================================================================
def safe_save_tiff(path):
    plt.savefig(path, dpi=300, format='tiff', bbox_inches='tight')
    print(f"✅ 图片已保存: {path}")

print("\n📊 绘制混淆矩阵...")
unique_labels = sorted(list(set(y_labels)))
label_map_display = {1: "Tolerant", -1: "Sensitive"}
display_labels = [label_map_display.get(l, f"Class_{l}") for l in unique_labels]

cm = confusion_matrix(y_labels, final_preds_list, labels=unique_labels)

cm_min = int(cm.min())
cm_max = int(cm.max())
cbar_ticks = list(np.linspace(cm_min, cm_max, num=5, dtype=int))
cbar_ticks = sorted(set(cbar_ticks))

fig, ax = plt.subplots(figsize=(20, 18))
heatmap = sns.heatmap(
    cm, annot=True, fmt='d', cmap='GnBu',
    xticklabels=display_labels,
    yticklabels=display_labels,
    annot_kws={"size": 70},
    cbar_kws={'ticks': cbar_ticks, 'shrink': 0.8, 'label': 'Count'},
    ax=ax
)
ax.set_xlabel(
    'Predicted Class',
    fontsize=60,
    fontweight='bold'
)

ax.set_ylabel(
    'Actual Class',
    fontsize=60,
    fontweight='bold'
)

# Tolerant / Sensitive字体
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0,
    fontsize=60,
    fontweight='bold'
)

ax.set_yticklabels(
    display_labels,
    rotation=90,
    fontsize=60,
    fontweight='bold',
    va='center'
)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=55)
cbar.set_label('Count', fontsize=55, fontweight='bold')
safe_save_tiff(os.path.join(output_dir, "Confusion_Matrix.tiff"))
plt.close()

# ==============================================================================
# 6. SHAP 分析（基于最终随机森林模型）
# ==============================================================================
print("\n🧬 生成 SHAP 解释图...")
try:
    # 在全量数据上标准化并训练最终随机森林模型（使用最优参数众数）
    final_scaler = StandardScaler()
    X_scaled_final = pd.DataFrame(
        final_scaler.fit_transform(X),
        index=X.index, columns=X.columns
    )

    final_model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        class_weight='balanced',
        random_state=42
    )
    final_model.fit(X_scaled_final, y_labels)

    # 使用 TreeExplainer（适配随机森林）
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_scaled_final)

    # 二分类时 shap_values 为 list，长度为 2，取正类（索引 1）
    if isinstance(shap_values, list):
        shap_vals_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        print(f"⚠ 二分类 SHAP，已取正类（index 1）的 shap_values")
    else:
        shap_vals_plot = shap_values

    plt.figure(figsize=(18, 14))
    shap.summary_plot(shap_vals_plot, X_scaled_final, max_display=10, show=False)
    plt.xlabel("SHAP value (impact on model output)", fontsize=FONT_SIZE_LABEL - 10)
    plt.tight_layout()
    safe_save_tiff(os.path.join(output_dir, "RF_SHAP_Summary_Plot.tiff"))
    plt.close()

    pd.DataFrame(shap_vals_plot, columns=X.columns, index=X.index).to_csv(
        os.path.join(output_dir, "Metabolite_SHAP_Values.csv"), encoding='utf-8-sig'
    )
except Exception as e:
    print(f"❌ SHAP 出错: {e}")

# ==============================================================================
# 7. 特征重要性排序（基于平均随机森林特征重要性）
# ==============================================================================
importances_array = np.array(all_feature_importances)   # 形状 = (总折数, 特征数)
avg_importance = np.mean(importances_array, axis=0)

feat_importance = pd.DataFrame({
    'Metabolite': X.columns,
    'Average_Importance': avg_importance,
}).sort_values(by='Average_Importance', ascending=False)

feat_importance.to_csv(
    os.path.join(output_dir, "Metabolite_Importance_Final_Rank.csv"),
    index=False, encoding='utf-8-sig'
)

print(f"\n✨ 全部任务完成！\n输出目录: {output_dir}")