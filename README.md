Classifying proteins into functional families can improve our understanding of a protein’s function and can allow transferring annotations within the same family. 
Toward this end, functional families need to be “pure”, i.e., contain only proteins with identical function. 
Functional Families (FunFams) [1] cluster proteins within CATH superfamilies [2] into such groups of proteins sharing function, based on differentially conserved residues.
11% of all FunFams (22,830 of 203,639) also contain EC annotations and of those, 7% (1,526 of 22,830) have at least two different EC annotations, i.e., inconsistent functional annotations.

We propose an approach to further cluster FunFams into smaller and functionally more consistent sub-families by encoding their sequences through embeddings. 
These embeddings originate from deep learned language models (LMs) transferring the knowledge gained from predicting missing amino acids in a sequence (ProtBERT [3]) and have been further optimized to distinguish between proteins belonging to the same or a different CATH superfamily (PB-Tucker [4]). 
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

# Example

`funfams_clustering_pipeline_example.py` can be executed to perform FunFams clustering for all FunFams in superfamily 3.20.20.70.


# Data

FunFams are accessible through CATH: `ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/`

ProtBERT and PB-Tucker embeddings can be obtained through [the bio_embeddings pipeline](https://github.com/sacdallago/bio_embeddings).

PB-Tucker embeddings for the 110,876 FunFams from 458 CATH superfamilies considered in this study can be accessed here:
* Embeddings:  `ftp://rostlab.org/FunFamsClustering/sequences_funfams_ec_all_protbert_tucker.npy`
* Corresponding ids: `ftp://rostlab.org/FunFamsClustering/sequences_funfams_ec_all_protbert_tucker_ids.txt`

EC annotations can be obtained through the UniProt SPARQL API.

Data set statistics considered in this study:
* List of all FunFams IDs in the 458 superfamilies: `data/funfams_ids.txt`
* List of all FunFams IDs with EC annotations in the 458 superfamilies: `data/funfams_ids_with_ecs.txt`
* Sequences and segments in FunFams v4.3: `ftp://rostlab.org/FunFamsClustering/sequences_funfams.txt`

# Cite

If you are using this method and find it helpful, we would appreciate if you could cite the following publication:

Littmann M, Bordin N, Heinzinger M, Orengo C, Rost B (2021). Clustering FunFams using sequence embeddings improves EC purity. Bioinformatics, 37(20).

# References
[1] Das S, Lee D, Sillitoe I, Dawson N, Lees JG, Orengo CA (2016). Functional classification of CATH superfamilies: a domain-based approach for protein function annotation. Bioinformatics, 32(18).

[2] Sillitoe I, Bordin N, Dawson N, Waman VP, Ashford P, et al (2021). CATH: increased structural coverage of functional space. Nucleic Acids Research, 49(D1).

[3] Elnaggar A, Heinzinger M, Dallago C, Rihawi G, Wang Y, et al (2021). ProtTrans: towards cracking the language of life's code through self-supervised deep learning and high performance computing. IEEE Transactions on Pattern Analysis and Machine Intelligence.

[4] Heinzinger M, Littmann M, Sillitoe I, Bordin N, Orengo C, Rost B (2022). Contrastive learning on protein embeddings enlightens midnight zone. NAR Genomics and Bioinformatics, 4(2).
