Classifying proteins into functional families can improve our understanding of a protein’s function and can allow transferring annotations within the same family. 
Toward this end, functional families need to be “pure”, i.e., contain only proteins with identical function. 
Functional Families (FunFams) cluster proteins within CATH su-perfamilies into such groups of proteins sharing function, based on differentially conserved residues.
11% of all FunFams (22,830 of 203,639) also contain EC annotations and of those, 7% (1,526 of 22,830) have at least two different EC annotations, i.e., inconsistent functional annotations.

We propose an approach to further cluster FunFams into smaller and functionally more consistent sub-families by encoding their sequences through embeddings. 
These embeddings originate from deep learned language models (LMs) transferring the knowledge gained from predicting missing amino acids in a sequence (ProtBERT) and have been further optimized to distinguish between proteins belonging to the same or a different CATH superfamily (PB-Tucker). 
Using distances between sequences in embedding space and DBSCAN to cluster FunFams, as well as identify outlier sequences, resulted in twice as many more pure clusters per FunFam than for a random clustering. 
52% of the impure FunFams were split into pure clusters, four times more than for random. 
While functional consistency was mainly measured using EC annotations, we observed similar results for binding annotations. 
Thus, we expect an increased purity also for other definitions of function. 
Our results can help generat-ing FunFams; the resulting clusters with improved functional consistency can be used to infer annotations more reliably. 
We expect this approach to succeed equally for any other grouping of proteins by their phenotypes.

# How to perform FunFams clustering

1. Calculate pairwise distances between all sequences within a FunFam for all FunFams in one superfamily
2. Determine distance threshold for clustering.
3. Perform clustering using this distance threshold and a user-defined neighborhood size (default: 5)
4. For FunFams with known EC annotations: Use EC annotations to validate approach

# Data

PB-Tucker embeddings for the 110,876 FunFams from 458 CATH superfamilies considered in this study can be accessed here:
* Embeddings:  ftp://rostlab.org/FunFamsClustering/sequences_funfams_ec_all_protbert_tucker.npy
* Corresponding ids: ftp://rostlab.org/FunFamsClustering/sequences_funfams_ec_all_protbert_tucker_ids.txt

# References
