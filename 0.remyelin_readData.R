#install.packages("qs")
library(qs)
library(Matrix)
library(tidyverse)

root_dir <- "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq"
puck_dirs <- list.dirs(root_dir, recursive = FALSE, full.names = TRUE)
puck_dirs = puck_dirs[grepl(pattern = '2023-08-25_Puck_*', x = puck_dirs)]
puck_list <- list()
for (puck_dir in puck_dirs) {
  # Find the .rds file inside the folder
  rds_file <- list.files(puck_dir, pattern = "final.rds", full.names = TRUE)
  
  if (length(rds_file) == 1) {
    puck_name <- basename(puck_dir)
    message("Reading: ", puck_name)
    
    # Read the RDS and store it in the list
    puck_list[[puck_name]] <- readRDS(rds_file)
  } else {
    warning("No unique .rds file found in ", puck_dir)
  }
}
## Warning messages: excluded samples?
#1: No unique .rds file found in /home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/2023-08-25_Puck_221207_28 
#2: No unique .rds file found in /home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/2023-08-25_Puck_221207_32 
#3: No unique .rds file found in /home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/2023-08-25_Puck_221207_35 

puck_name = "2023-08-25_Puck_230117_01"
puck = puck_list[[puck_name]]
hist(puck$nCount_Spatial)
hist(puck$nFeature_Spatial)

###### importing the merged and processed datasets
pucks_merged <- qread("/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline.qs")
head(pucks_merged@meta.data)

##### saving to import to python env
### leads to error?
# SaveH5Seurat(object = pucks_merged, filename = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline.h5Seurat")

pucks_merged <- qread("/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline.qs")
######################################################################
##### saving the scaled data - includes negative values -> inconsistent with NMF
scaled_mat <- Matrix(pucks_merged@assays$Spatial$scale.data, sparse = TRUE)
scaled_mat <- Matrix(pucks_merged@assays$Spatial$counts.1, sparse = TRUE)
writeMM(scaled_mat, file = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline/all_final_cropped_pucks_standardpipeline_scaled_counts.mtx")
write.table(rownames(scaled_mat), file = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline/all_final_cropped_pucks_standardpipeline_features.tsv", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(colnames(pucks_merged), file = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline/all_final_cropped_pucks_standardpipeline_barcodes.tsv", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.csv(pucks_merged@meta.data, file = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline/all_final_cropped_pucks_standardpipeline_metadata.csv")

######################################################################
##### saving the merged count data
count_layer_names <- grep("^counts\\.[0-9]+$", Layers(pucks_merged[["Spatial"]]), value = TRUE)
count_list <- lapply(count_layer_names, function(layer) {
  GetAssayData(pucks_merged, assay = "Spatial", layer = layer)
})
lapply(count_list, dim) ### each puck has a different number of genes
shared_genes <- Reduce(intersect, lapply(count_list, rownames))
length(shared_genes) 

count_list_shared <- lapply(count_list, function(mat) {
  mat[shared_genes, , drop = FALSE]
})
counts_merged <- do.call(cbind, count_list_shared)

writeMM(counts_merged, file = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline/all_final_cropped_pucks_standardpipeline_counts_merged.mtx")
write.table(rownames(counts_merged), file = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline/all_final_cropped_pucks_standardpipeline_features.tsv", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(colnames(counts_merged), file = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline/all_final_cropped_pucks_standardpipeline_barcodes.tsv", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.csv(pucks_merged@meta.data, file = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline/all_final_cropped_pucks_standardpipeline_metadata.csv")

sum(rownames(pucks_merged@meta.data) != colnames(counts_merged))
