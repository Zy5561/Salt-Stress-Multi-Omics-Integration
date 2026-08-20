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
from sklearn.linear_model import LogisticRegression   # 替换 SVC
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
metabolite_path = r"Metabolite_Matrix.csv"
label_path = r"耐盐性划分.xlsx"
output_dir = r"LR"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_met = pd.read_csv(metabolite_path, index_col=0, encoding='utf-8-sig')
df_label = pd.read_excel(label_path, index_col=0)

df_merged = df_met.join(df_label.iloc[:, 0], how='inner')
y_labels = df_merged.iloc[:, -1].astype(int)
X = df_merged.iloc[:, :-1].fillna(0)

print(f"✅ 数据读取完成。样本量: {len(X)}, 特征数: {X.shape[1]}, 标签类别: {set(y_labels)}")

# ==============================================================================
# 3. 嵌套交叉验证建模（逻辑回归）
# ==============================================================================
# 逻辑回归参数网格：只调整 C，使用 L2 正则化，liblinear 求解器支持 class_weight 和 probability
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
n_repeats = 100

prediction_storage = {idx: [] for idx in X.index}
all_coefs = []
all_best_C = []

print("🔄 正在执行嵌套交叉验证（100次重复，逻辑回归模型）...")

for r in range(n_repeats):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r * 123)

    for train_idx, test_idx in outer_cv.split(X, y_labels):
        X_train_raw = X.iloc[train_idx]
        X_test_raw  = X.iloc[test_idx]
        y_train = y_labels.iloc[train_idx]
        y_test  = y_labels.iloc[test_idx]

        # 标准化（每次 CV 内独立 fit，避免数据泄露）
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            index=X_train_raw.index, columns=X_train_raw.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_raw),
            index=X_test_raw.index, columns=X_test_raw.columns
        )

        # 逻辑回归模型（与 SVM 线性核类似，但概率输出更自然）
        grid_search = GridSearchCV(
            LogisticRegression(
                penalty='l2',
                class_weight='balanced',
                solver='liblinear',   # 支持概率预测和 class_weight
                random_state=r * 7 + 3,
                max_iter=1000
            ),
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=r * 7 + 3),
            scoring='balanced_accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        all_best_C.append(grid_search.best_params_['C'])

        preds = model.predict(X_test)
        for i, idx in enumerate(X_test.index):
            prediction_storage[idx].append(preds[i])

        # 存储 C * coef_ 以统一不同 C 值下的量纲，便于后续平均
        all_coefs.append(model.coef_[0] * model.C)

    if (r + 1) % 20 == 0:
        print(f"  ✔ 已完成 {r + 1}/{n_repeats}")

# 多数投票确定最终预测
final_preds_list = []
for idx in X.index:
    votes = prediction_storage[idx]
    most_common_label = Counter(votes).most_common(1)[0][0]
    final_preds_list.append(most_common_label)

# 最优 C 的众数（用于最终全量模型）
best_C_mode = Counter(all_best_C).most_common(1)[0][0]
print(f"\n📊 100×5 折 CV 中最优 C 众数: {best_C_mode}")
print(f"   C 值分布: {Counter(all_best_C)}")

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
cbar.ax.tick_params(labelsize=60)
cbar.set_label('Count', fontsize=55, fontweight='bold')
safe_save_tiff(os.path.join(output_dir, "Confusion_Matrix.tiff"))
plt.close()

# ==============================================================================
# 6. SHAP 分析（基于最终逻辑回归模型）
# ==============================================================================
print("\n🧬 生成 SHAP 解释图...")
try:
    # 在全量数据上标准化并训练最终逻辑回归模型（使用最优 C 众数）
    final_scaler = StandardScaler()
    X_scaled_final = pd.DataFrame(
        final_scaler.fit_transform(X),
        index=X.index, columns=X.columns
    )

    final_model = LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        C=best_C_mode,
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    final_model.fit(X_scaled_final, y_labels)

    # LinearExplainer 适用于线性模型（逻辑回归）
    explainer = shap.LinearExplainer(final_model, X_scaled_final)
    shap_values = explainer.shap_values(X_scaled_final)

    # 处理 SHAP 输出（二分类时可能返回列表）
    if isinstance(shap_values, list):
        shap_vals_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        print(f"⚠ 多分类 SHAP，已取正类（index 1）的 shap_values")
    else:
        shap_vals_plot = shap_values

    plt.figure(figsize=(18, 14))
    shap.summary_plot(shap_vals_plot, X_scaled_final, max_display=10, show=False)
    plt.xlabel("SHAP value (impact on model output)", fontsize=FONT_SIZE_LABEL - 10)
    plt.tight_layout()
    safe_save_tiff(os.path.join(output_dir, "Logistic_SHAP_Summary_Plot.tiff"))
    plt.close()

    pd.DataFrame(shap_vals_plot, columns=X.columns, index=X.index).to_csv(
        os.path.join(output_dir, "Metabolite_SHAP_Values.csv"), encoding='utf-8-sig'
    )
except Exception as e:
    print(f"❌ SHAP 出错: {e}")

# ==============================================================================
# 7. 特征重要性排序（基于平均 |C * coef_|）
# ==============================================================================
coefs_array = np.array(all_coefs)       # 形状 = (总折数, 特征数)
avg_weight = np.mean(coefs_array, axis=0)
abs_avg_weight = np.mean(np.abs(coefs_array), axis=0)

feat_importance = pd.DataFrame({
    'Metabolite': X.columns,
    'Average_Weight': avg_weight,
    'Abs_Average_Weight': abs_avg_weight,
}).sort_values(by='Abs_Average_Weight', ascending=False)

feat_importance.to_csv(
    os.path.join(output_dir, "Metabolite_Importance_Final_Rank.csv"),
    index=False, encoding='utf-8-sig'
)

print(f"\n✨ 全部任务完成！\n输出目录: {output_dir}")