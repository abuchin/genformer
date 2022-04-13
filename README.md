# genformer dev repository

**genformer** predicts rna-seq from atac-seq + sequence

## dataset inputs
See https://app.terra.bio/#workspaces/epigenomics/gro_nn for data processing workflows
and input data. Current version takes in inputs of length 409600 basepairs and outputs a (50,) tensor,
representing the RNA-seq signal at 4096 bp resolution. The RNA-seq signal is the log(1+TPM)
of the gene(s) who's TSS overlies a given bin

## initial architecture
Mainly follows Enformer architecture:
 * 6 convolutional layers
 * 2-4 transformer blocks w/ linear performer attention
 * cropping layer
 * final conv, averaging over channel dimension, gelu, final dense

## training

Define hyper- and sweep parameters in train_model.py Use execute.sh
to execute sweep on a v3-8 TPU and log results to an input wandb project.

