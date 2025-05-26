#!/usr/bin/env python

######################## Plotting the results
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import tensorflow as tf
from spatial_factorization import SpatialFactorization, ModelTrainer
from scipy import sparse
import os
import time

#help(spatial_factorization.ModelTrainer)
#help(spatial_factorization.SpatialFactorization)
'''
## Load the merged data and split by sample
adata = sc.read_h5ad("/home/delaram/LabelCorrection/data_PSC/PSC_merged.h5ad")
sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
                   for sample_id in sample_ids}
adata_normal_A = adata_by_sample['normal_A']
'''
### Load the normal_A data
adata_normal_A = sc.read('../data_PSC/normal_A.h5ad')

Y = adata_normal_A.X
if sparse.issparse(Y):
    Y = Y.toarray()
Y = Y.astype("float32")

Z = adata_normal_A.obsm['spatial']
Z = Z.astype("float32")

# Dimensions
J = Y.shape[1]
L = 10
Ntr = Y.shape[0]  # total number of training examples

# Fake size factors (use row sums or other relevant normalization if needed)
#sz = Y.sum(axis=1).reshape(-1, 1).astype("float32")
sz = np.ones((Y.shape[0], 1), dtype="float32")

# Number of inducing points (here same as input, but can be subsampled)
Z_inducing = Z.copy()

# Initialize model
#model = SpatialFactorization(J=J, L=L, Z=Z_inducing, lik='poi')
model = SpatialFactorization(J=J, L=10, Z=Z, lik='gau', disp=1.0)
trainer = ModelTrainer(model)

# Create TensorFlow dataset of (X, Y, sz)
# Convert (X, Y, sz) tuple to dict {"X": X, "Y": Y, "sz": sz}
dataset = tf.data.Dataset.from_tensor_slices((Z, Y, sz)).map(
    lambda x, y, z: {"X": x, "Y": y, "sz": z}
).batch(256)


# Create a custom save path
save_dir = "/home/delaram/LabelCorrection/results/model_checkpoints"
# Make sure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Generate and set custom pickle path
pickle_path = model.generate_pickle_path(sz="none", base=save_dir)
trainer.set_pickle_path(pickle_path)

### start the timer
start_time = time.time()

trainer.train_model(
    dataset,
    Ntr=Ntr,
    S=1,
    verbose=True
)

### end the timer
end_time = time.time()

print(f"Training time: {end_time - start_time} seconds")
# Save the model
trainer.pickle(0.0, 0.0)
pd.DataFrame(trainer.loss).to_csv(os.path.join(save_dir, "training_loss.csv"), index=False)