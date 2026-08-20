library(igraph)
library(ggraph)
library(ggplot2)

outdir <- "D:/试验/试验二（代谢+转录组综合评估）/第二篇文章数据整理(final-4.25)/WGCNA分析-5.18-final/Result/green"

edges <- read.delim(paste0(outdir, "/Cytoscape_edges_green.txt"),
                    header=TRUE, stringsAsFactors=FALSE, check.names=FALSE)
nodes <- read.delim(paste0(outdir, "/Cytoscape_nodes_green.txt"),
                    header=TRUE, stringsAsFactors=FALSE, check.names=FALSE)

g         <- graph_from_data_frame(d=edges, vertices=nodes, directed=FALSE)
deg       <- degree(g)
hub_genes <- names(sort(deg, decreasing=TRUE))[1:10]

# 输出Hub基因名称
cat("========================================\n")
cat("Top 10 Hub genes (by degree):\n")
print(data.frame(
  Rank   = 1:10,
  Gene   = hub_genes,
  Degree = deg[hub_genes]
))
cat("========================================\n")

hub_ids       <- which(V(g)$name %in% hub_genes)
hub_neighbors <- unique(unlist(lapply(hub_ids, function(x) neighbors(g, x))))
hub_subnet    <- induced_subgraph(g, vids=unique(c(hub_ids, hub_neighbors)))

w_all       <- E(hub_subnet)$weight
w_threshold <- quantile(w_all, 0.95)
hub_slim    <- delete_edges(hub_subnet, which(w_all < w_threshold))
cat("过滤后边数:", ecount(hub_slim), "\n")

is_hub        <- V(hub_slim)$name %in% hub_genes
hub_name_set  <- V(hub_slim)$name[is_hub]
non_hub_names <- V(hub_slim)$name[!is_hub]

hub_conn_count <- sapply(non_hub_names, function(vname) {
  nbrs <- neighbors(hub_slim, vname)
  sum(V(hub_slim)$name[nbrs] %in% hub_name_set)
})

middle_nodes <- non_hub_names[hub_conn_count >= 2]
outer_nodes  <- non_hub_names[hub_conn_count < 2]

cat("各层节点数 — Hub:", length(hub_name_set),
    " | 中圈:", length(middle_nodes),
    " | 外圈:", length(outer_nodes), "\n")

place_on_circle <- function(nms, radius, offset_angle=0) {
  n      <- length(nms)
  if (n == 0) return(data.frame(name=character(0), x=numeric(0), y=numeric(0)))
  angles <- seq(0, 2*pi, length.out=n+1)[-(n+1)] + offset_angle
  data.frame(name=nms, x=radius*cos(angles), y=radius*sin(angles))
}

set.seed(42)
layout_df <- rbind(
  place_on_circle(sample(hub_name_set),  0.18, pi/2),
  place_on_circle(sample(middle_nodes),  0.52, pi/6),
  place_on_circle(sample(outer_nodes),   0.88, 0)
)

layout_df  <- layout_df[match(V(hub_slim)$name, layout_df$name), ]
layout_mat <- as.matrix(layout_df[, c("x","y")])

deg_sub   <- degree(hub_slim)
size_norm <- (deg_sub - min(deg_sub)) / (max(deg_sub) - min(deg_sub) + 1e-6)

V(hub_slim)$layer <- ifelse(is_hub, "Hub",
                            ifelse(V(hub_slim)$name %in% middle_nodes, "Middle", "Outer"))
V(hub_slim)$vsize <- ifelse(is_hub, 5 + size_norm*3,
                            ifelse(V(hub_slim)$name %in% middle_nodes, 2 + size_norm*1.5, 1.2))

w_slim         <- E(hub_slim)$weight
wn             <- (w_slim - min(w_slim)) / (max(w_slim) - min(w_slim) + 1e-6)
E(hub_slim)$wn <- wn

hub_idx      <- which(V(hub_slim)$name %in% hub_genes)
hub_coords   <- layout_mat[hub_idx, ]
label_offset <- 1.22
hub_label_x  <- hub_coords[,1] * label_offset
hub_label_y  <- hub_coords[,2] * label_offset

p <- ggraph(hub_slim, layout=layout_mat) +
  
  annotate("path",
           x=0.18*cos(seq(0,2*pi,length.out=300)),
           y=0.18*sin(seq(0,2*pi,length.out=300)),
           colour="#5B3FA0", alpha=0.5, linewidth=0.6, linetype="dashed") +
  annotate("path",
           x=0.52*cos(seq(0,2*pi,length.out=300)),
           y=0.52*sin(seq(0,2*pi,length.out=300)),
           colour="#1DAE8F", alpha=0.45, linewidth=0.5, linetype="dashed") +
  annotate("path",
           x=0.88*cos(seq(0,2*pi,length.out=300)),
           y=0.88*sin(seq(0,2*pi,length.out=300)),
           colour="#72BE44", alpha=0.4, linewidth=0.45, linetype="dashed") +
  
  geom_edge_link(aes(alpha=wn, width=wn),
                 colour="grey50", show.legend=FALSE) +
  scale_edge_alpha(range=c(0.02, 0.18)) +
  scale_edge_width(range=c(0.05, 0.5)) +
  
  geom_node_point(
    data = function(x) x[x$layer != "Hub", ],
    aes(size=vsize, colour=layer, fill=layer),
    shape=21, stroke=0.4, show.legend=FALSE
  ) +
  
  geom_node_point(
    data = function(x) x[x$layer == "Hub", ],
    aes(size=vsize),
    shape=21, fill="#5B3FA0", colour="white",
    stroke=1.2, show.legend=FALSE
  ) +
  
  scale_colour_manual(values=c(Hub="#3A2570", Middle="#0F7A65", Outer="#4A9020")) +
  scale_fill_manual(values=c(Hub="#5B3FA0", Middle="#1DAE8F", Outer="#72BE44")) +
  scale_size_identity() +
  
  annotate("text",
           x=hub_label_x, y=hub_label_y,
           label=V(hub_slim)$name[hub_idx],
           size=4.5, fontface="bold",
           colour="#2C1654",
           hjust=ifelse(hub_label_x > 0, 0, 1)) +
  
  labs(title="Hub gene co-expression network (green module)") +
  scale_x_continuous(expand=expansion(0)) +
  scale_y_continuous(expand=expansion(0)) +
  coord_fixed(xlim=c(-0.92, 0.92), ylim=c(-0.92, 0.92), clip="on") +
  theme_void() +
  theme(
    plot.title      = element_text(size=14, face="bold", colour="#2C2C2C",
                                   hjust=0.5, margin=margin(b=4)),
    plot.background = element_rect(fill="white", colour=NA),
    plot.margin     = margin(2, 2, 2, 2),
    legend.position = "none"
  )

ggsave(paste0(outdir, "/Hub_network_turquoise_concentric_v5.tiff"),
       plot=p, width=6, height=6, dpi=300, compression="lzw")

# 同时保存Hub基因列表到CSV
write.csv(
  data.frame(Rank=1:10, Gene=hub_genes, Degree=deg[hub_genes]),
  paste0(outdir, "/Hub_genes_turquoise.csv"),
  row.names=FALSE
)

cat("✅ 完成！图片和Hub基因列表已保存。\n")