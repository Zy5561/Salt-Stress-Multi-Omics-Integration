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
from sklearn.svm import SVC
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

# ━━━ 修改1 START ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 原代码：在全量 X 上 fit_transform，再把 X_scaled 传入 CV，导致数据泄露。
# 修改：删除此处的标准化，改为在每次 CV 内部对 X_train/X_test 分别 fit/transform。
# ━━━ 修改1 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ==============================================================================
# 3. 嵌套交叉验证建模
# ==============================================================================
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
n_repeats = 100

prediction_storage = {idx: [] for idx in X.index}
all_coefs = []

# ━━━ 修改4 START ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 新增：记录每次 CV 折的最优 C 值，用于事后统计众数并设置 SHAP 模型
all_best_C = []
# ━━━ 修改4 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("🔄 正在执行嵌套交叉验证（100次重复）...")

for r in range(n_repeats):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r * 123)

    for train_idx, test_idx in outer_cv.split(X, y_labels):  # ← 用原始 X
        X_train_raw = X.iloc[train_idx]
        X_test_raw  = X.iloc[test_idx]
        y_train = y_labels.iloc[train_idx]
        y_test  = y_labels.iloc[test_idx]

        # ━━━ 修改1（续）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 标准化只在训练折内 fit，再 transform 测试折，避免数据泄露
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            index=X_train_raw.index, columns=X_train_raw.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_raw),
            index=X_test_raw.index, columns=X_test_raw.columns
        )
        # ━━━ 修改1 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        grid_search = GridSearchCV(
            SVC(kernel='linear', class_weight='balanced', probability=True),
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=r * 7 + 3),
            scoring='balanced_accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        # ━━━ 修改4（续）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        all_best_C.append(grid_search.best_params_['C'])
        # ━━━ 修改4 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        preds = model.predict(X_test)
        for i, idx in enumerate(X_test.index):
            prediction_storage[idx].append(preds[i])

        # ━━━ 修改3 START ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 原代码：直接存 coef_[0]（不同 C 值下系数量纲不可比）
        # 修改：存储 C * coef_（即间隔向量的 support 贡献），量纲统一后再平均
        all_coefs.append(model.coef_[0] * model.C)
        # ━━━ 修改3 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if (r + 1) % 20 == 0:
        print(f"  ✔ 已完成 {r + 1}/{n_repeats}")

# 多数投票
final_preds_list = []
for idx in X.index:
    votes = prediction_storage[idx]
    most_common_label = Counter(votes).most_common(1)[0][0]
    final_preds_list.append(most_common_label)

# ━━━ 修改4（续）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 统计最优 C 的众数，用于 SHAP 最终模型
best_C_mode = Counter(all_best_C).most_common(1)[0][0]
print(f"\n📊 100×5 折 CV 中最优 C 众数: {best_C_mode}")
print(f"   C 值分布: {Counter(all_best_C)}")
# ━━━ 修改4 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ==============================================================================
# 4. 导出品种预测明细表（逻辑不变）
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

# ━━━ 修改6 START ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 原代码：步长用整除可能产生过密刻度；改为固定生成 5 个刻度，自动适应范围
cbar_ticks = list(np.linspace(cm_min, cm_max, num=5, dtype=int))
cbar_ticks = sorted(set(cbar_ticks))  # 去重（防止 min==max 时重复）
# ━━━ 修改6 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
# 6. SHAP 分析
# ==============================================================================
print("\n🧬 生成 SHAP 解释图...")
try:
    # ━━━ 修改1 & 修改2 START ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 修改1：SHAP 最终模型也需要在全量 X 上重新 fit scaler（不能用 CV 内部的 scaler）
    # 修改2：C 值改用 CV 统计的众数，不再硬编码
    final_scaler = StandardScaler()
    X_scaled_final = pd.DataFrame(
        final_scaler.fit_transform(X),
        index=X.index, columns=X.columns
    )

    final_model = SVC(
        kernel='linear',
        class_weight='balanced',
        probability=True,
        C=best_C_mode,   # ← 修改2：从 CV 众数获取，不再硬编码 C=0.1
        random_state=42
    )
    final_model.fit(X_scaled_final, y_labels)
    # ━━━ 修改1 & 修改2 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    explainer = shap.LinearExplainer(final_model, X_scaled_final)
    shap_values = explainer.shap_values(X_scaled_final)

    # ━━━ 修改5 START ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 原代码：直接使用 shap_values，二分类时 LinearExplainer 返回形状为 (n, p) 无需处理
    # 修改：显式确认维度，多分类时取正类对应的 shap_values
    if isinstance(shap_values, list):
        # 多分类：取最大类 index（或正类 index 1）
        shap_vals_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        print(f"⚠ 多分类 SHAP，已取 class index 1 的 shap_values")
    else:
        shap_vals_plot = shap_values
    # ━━━ 修改5 END ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    plt.figure(figsize=(18, 14))
    shap.summary_plot(shap_vals_plot, X_scaled_final, max_display=10, show=False)
    plt.xlabel("SHAP value (impact on model output)", fontsize=FONT_SIZE_LABEL - 10)
    plt.tight_layout()
    safe_save_tiff(os.path.join(output_dir, "SVM_SHAP_Summary_Plot.tiff"))
    plt.close()

    pd.DataFrame(shap_vals_plot, columns=X.columns, index=X.index).to_csv(
        os.path.join(output_dir, "Metabolite_SHAP_Values.csv"), encoding='utf-8-sig'
    )
except Exception as e:
    print(f"❌ SHAP 出错: {e}")

# ==============================================================================
# 7. 特征重要性排序
# ==============================================================================
coefs_array = np.array(all_coefs)       # 已经是 C * coef_，量纲统一
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