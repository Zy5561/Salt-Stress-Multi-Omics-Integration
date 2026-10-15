# ======================================================
# K-Means Clustering Analysis + Export Cluster Labels
# Only TIFF, ULTRA-LARGE fonts (no bold), small dots, no edges
# Cluster 1 → Salt sensitive, Cluster 2 → Salt tolerant
# ======================================================

import os
os.environ["OMP_NUM_THREADS"] = "2"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# -----------------------------
# 基础图形风格（其他图字号也适度提高）
# -----------------------------
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 30
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["axes.titlesize"] = 30
plt.rcParams["legend.fontsize"] = 28
plt.rcParams["xtick.labelsize"] = 26
plt.rcParams["ytick.labelsize"] = 26

# ============================================================
# 【修改】设置输出目录为你的指定文件夹
# ============================================================
output_dir = r"D:\没错，是毕业论文啊\毕业了论文框架\第三章-多组学\丁老师指导后数据处理\数据整理-final-8月\300个品种高光谱+形态指标数据采集（V2）"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   # 如果目录不存在则创建（以防万一）

# -----------------------------
# 读取数据（输入文件也在此目录下）
# -----------------------------
file = os.path.join(output_dir, "300个品种根长、芽长、生理指标-final.xlsx")
df = pd.read_excel(file)
sample_name = df.iloc[:, 0]
X = df.iloc[:, 1:]
print("Data shape:", X.shape)

# -----------------------------
# 标准化
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 计算最佳K
# -----------------------------
K = range(2, 11)
SSE, SIL, CH, DB = [], [], [], []
print("\nEvaluation index")
for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    label = model.fit_predict(X_scaled)
    SSE.append(model.inertia_)
    SIL.append(silhouette_score(X_scaled, label))
    CH.append(calinski_harabasz_score(X_scaled, label))
    DB.append(davies_bouldin_score(X_scaled, label))
    print(f"K={k}  SIL={SIL[-1]:.3f}  CH={CH[-1]:.2f}  DB={DB[-1]:.3f}")

best_k = K[np.argmax(SIL)]
print("\nBest K =", best_k)

# -----------------------------
# Figure1 Elbow (TIFF only)
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(list(K), SSE, marker="o", linewidth=2)
plt.xlabel("Number of clusters (K)")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.tight_layout()
fig1_path = os.path.join(output_dir, "Figure1_Elbow.tif")
plt.savefig(fig1_path, dpi=600)
print(f"Figure 1 saved to: {os.path.abspath(fig1_path)}")
plt.close()

# -----------------------------
# Figure2 Silhouette (TIFF only)
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(list(K), SIL, marker="o", linewidth=2)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.tight_layout()
fig2_path = os.path.join(output_dir, "Figure2_Silhouette.tif")
plt.savefig(fig2_path, dpi=600)
print(f"Figure 2 saved to: {os.path.abspath(fig2_path)}")
plt.close()

# -----------------------------
# 最终聚类
# -----------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster = kmeans.fit_predict(X_scaled)

# ---------- 定义聚类名称 ----------
cluster_display_names = []
for i in range(best_k):
    if i == 0:
        cluster_display_names.append("Salt sensitive")
    elif i == 1:
        cluster_display_names.append("Salt tolerant")
    else:
        cluster_display_names.append(f"Cluster {i+1}")

print("\nCluster size")
for i in range(best_k):
    print(cluster_display_names[i], np.sum(cluster == i))

# -----------------------------
# 导出每个品种的聚类结果（保存到输出目录）
# -----------------------------
result_df = pd.DataFrame({'品种': sample_name, '所属类别': cluster + 1})
excel_path = os.path.join(output_dir, '品种聚类结果.xlsx')
result_df.to_excel(excel_path, index=False)
print(f"Cluster labels saved to: {os.path.abspath(excel_path)}")

# ================================================================
# Figure3 PCA —— 超大字体，散点放小、无黑边，且全部不加粗（完美铺满画布）
# ================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 超大画布
fig, ax = plt.subplots(figsize=(26, 23))

custom_colors = ["#1f77b4", "#E8694A"]
color_list = [custom_colors[i % len(custom_colors)] for i in range(best_k)]

# 散点放小，无黑边
for i in range(best_k):
    ax.scatter(
        X_pca[cluster == i, 0],
        X_pca[cluster == i, 1],
        s=120,
        color=color_list[i],
        label=cluster_display_names[i]
    )

# 轴标签：100号，不加粗
ax.set_xlabel(
    f"PC1 ({100*pca.explained_variance_ratio_[0]:.2f}%)",
    fontsize=80,
    fontweight='normal',
    labelpad=20
)
ax.set_ylabel(
    f"PC2 ({100*pca.explained_variance_ratio_[1]:.2f}%)",
    fontsize=80,
    fontweight='normal',
    labelpad=20
)

# 刻度字体：90号，不加粗
ax.tick_params(
    axis='both',
    labelsize=90,
    width=4.0,
    length=18
)

# 坐标轴线宽加粗
for spine in ax.spines.values():
    spine.set_linewidth(4.0)

# 图例：60号，不加粗
legend = ax.legend(
    frameon=False,
    fontsize=80,
    markerscale=2.5,
    handletextpad=1.0,
    loc="upper right"
)
for text in legend.get_texts():
    text.set_fontweight('normal')

plt.tight_layout(pad=0.5)
fig3_path = os.path.join(output_dir, "Figure3_PCA.tif")
plt.savefig(
    fig3_path,
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"}
)
print(f"Figure 3 saved to: {os.path.abspath(fig3_path)}")
plt.close()

# -----------------------------
# Cluster中心热图 (同步优化为大图 TIFF)
# -----------------------------
center = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(center, cmap="RdBu_r", aspect="auto")

ax.set_xticks(np.arange(len(X.columns)))
ax.set_xticklabels(X.columns, rotation=45, ha="right", fontsize=24)
ax.set_yticks(np.arange(best_k))
ax.set_yticklabels(cluster_display_names, fontsize=24)

cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Scaled value", fontsize=24)

plt.tight_layout()
fig4_path = os.path.join(output_dir, "Figure4_Heatmap.tif")
plt.savefig(
    fig4_path,
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"}
)
print(f"Figure 4 saved to: {os.path.abspath(fig4_path)}")
plt.close()

print("\n===================================")
print("Analysis finished!")
print(f"All files saved to: {output_dir}")
print("===================================")