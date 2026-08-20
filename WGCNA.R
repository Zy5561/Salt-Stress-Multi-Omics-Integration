## =========================================================
## 0. 加载包 + 环境配置
## =========================================================
library(WGCNA)
library(limma)  # 用于可能的批次校正
options(stringsAsFactors = FALSE)
enableWGCNAThreads() # 开启多线程加速

# 输出目录
outdir <- "D:/试验/试验二（代谢+转录组综合评估）/第二篇文章数据整理(final-4.25)/WGCNA分析/个人数据4.28"
if (!dir.exists(outdir)) {
  dir.create(outdir, recursive = TRUE)
}

## =========================================================
## 1. 读取数据
## =========================================================
# 基因表达矩阵：行为基因，列为样本
gene_exp <- read.csv("D:/试验/试验二（代谢+转录组综合评估）/第二篇文章数据整理(final-4.25)/WGCNA分析/个人数据4.28/LiverFemale-all.csv", row.names = 1)
# 性状数据
sample_info <- read.csv("D:/试验/试验二（代谢+转录组综合评估）/第二篇文章数据整理(final-4.25)/WGCNA分析/个人数据4.28/ClinicalTraits.csv", row.names = 1)
# 基因注释
gene_info <- read.csv("D:/试验/试验二（代谢+转录组综合评估）/第二篇文章数据整理(final-4.25)/WGCNA分析/个人数据4.28/GeneAnnotation.csv")

## =========================================================
## 2. 数据预处理
## =========================================================
datExpr0 <- t(gene_exp) # 转置为：行为样本，列为基因

# 检查缺失值过多的基因和样本
gsg <- goodSamplesGenes(datExpr0, minFraction = 0.5)
if (!gsg$allOK) {
  datExpr0 = datExpr0[gsg$goodSamples, gsg$goodGenes]
}
datExpr <- datExpr0

# 样本聚类图：检查是否有明显离群样本
tiff(paste0(outdir, "/Sample_clustering.tiff"), width = 2000, height = 1600, res = 300)
sampleTree <- hclust(dist(datExpr), method = "average")
plot(sampleTree, main = "Sample clustering", sub = "", xlab = "", cex.lab = 1.5, cex.axis = 1.5)
dev.off()

## =========================================================
## 3. 性状数据对齐
## =========================================================
datTraits <- sample_info[rownames(datExpr), ]

## =========================================================
## 4. 软阈值分析 (Soft Thresholding)
## =========================================================
powers <- 1:20
sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)

# 保存软阈值选择图
tiff(paste0(outdir, "/Soft_threshold_QC.tiff"), width = 2400, height = 1200, res = 300)
par(mfrow = c(1,2))
# 左图：无尺度独立性 (Scale Independence)
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)", ylab="Scale Free Topology Model Fit,signed R^2",
     type="n", main = "Scale independence")
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers, col="red")
abline(h=0.8, col="red")
# 右图：平均连通度 (Mean Connectivity)
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)", ylab="Mean Connectivity",
     type="n", main = "Mean connectivity")
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, col="red")
dev.off()

# 【关键修改】手动指定 Power = 16 (基于你之前的 QC 图)
softPower <- 16 

## =========================================================
## 5. 构建网络与模块识别 (一步法)
## =========================================================
# 【关键修改】增加 maxBlockSize 以防止基因被拆分成多个 Model/Block
net <- blockwiseModules(
  datExpr,
  power = softPower,
  maxBlockSize = 20000,      # 必须大于你的基因总数 (31579)
  TOMType = "unsigned",      # 保持与 pickSoftThreshold 一致
  networkType = "unsigned",
  minModuleSize = 30,
  mergeCutHeight = 0.25,     # 模块合并阈值
  numericLabels = FALSE,     # 输出颜色标签而非数字
  pamRespectsDendro = FALSE,
  saveTOMs = TRUE,
  saveTOMFileBase = paste0(outdir, "/TOM"),
  verbose = 3
)

# 输出模块大小统计
write.csv(table(net$colors), paste0(outdir, "/Module_size_summary.csv"))

## =========================================================
## 6. 模块树图 (可视化)
## =========================================================
tiff(paste0(outdir, "/Module_dendrogram_Final.tiff"), width = 2400, height = 1600, res = 300)

# 确保颜色与树的对象长度一致 (使用 blockGenes 索引)
plotDendroAndColors(
  net$dendrograms[[1]], 
  net$colors[net$blockGenes[[1]]],
  "Module colors",
  dendroLabels = FALSE, 
  hang = 0.03,
  addGuide = TRUE
)
dev.off()

## =========================================================
## 7. 模块-性状关联分析
## =========================================================
MEs <- net$MEs
# 计算模块特征向量与性状的相关性 (使用 Spearman 更稳健)
moduleTraitCor <- cor(MEs, datTraits, use = "p", method = "pearson")
moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nrow(datExpr))

# 绘制热图
tiff(paste0(outdir, "/Module_trait_relationship_Heatmap.tiff"), width = 2400, height = 2000, res = 300)

textMatrix <- paste(signif(moduleTraitCor, 2), "\n(", signif(moduleTraitPvalue, 1), ")", sep = "")
dim(textMatrix) <- dim(moduleTraitCor)

labeledHeatmap(
  Matrix = moduleTraitCor,
  xLabels = colnames(datTraits),
  yLabels = colnames(MEs),
  ySymbols = colnames(MEs),
  colorLabels = FALSE,
  colors = blueWhiteRed(50),
  textMatrix = textMatrix,
  setStdMargins = FALSE,
  cex.text = 0.6,
  zlim = c(-1,1),
  main = "Module-trait relationships"
)
dev.off()

## =========================================================
## 8. 导出基因列表与注释
## =========================================================
wgcna_result <- data.frame(gene_id = colnames(datExpr), module = net$colors)
wgcna_result <- merge(wgcna_result, gene_info, by = "gene_id", all.x = TRUE)
write.csv(wgcna_result, paste0(outdir, "/WGCNA_Final_Gene_Assignment.csv"), row.names = FALSE)

print("WGCNA 分析全流程运行完毕！")