from file_manager import FileManager, Npy2Npz
from distance_calculator import EmbeddingDistance
from determine_dist_cutoff import CutoffDetermination
from clustering_pipeline import DBSCANClustering

import os
import numpy as np


def main():

    superfamily = '3.20.20.70'

    # Read input
    # Embeddings for all 458 superfamilies considered in this study can be downloaded from
    # ftp://rostlab.org/FunFamsClustering/sequences_funfams_ec_all_protbert_tucker.npy
    embeddings_in = 'path_to_embeddings/sequences_funfams_ecs_all_protbert_tucker.npy'
    embedding_ids_in = 'path_to_embeddings/sequences_funfams_ecs_all_protbert_tucker_ids.txt'
    embeddings_tucker = Npy2Npz.get_dataset_uncompressed(embeddings_in, embedding_ids_in)

    # Read FunFam ids into a dictionary with key: superfamily, value: FunFams
    funfams = FileManager.read_funfam_ids_with_family('example/funfams_ids.txt')

    # Read sequence information per FunFam
    sequence_info_in = 'example/sequences_funfams.txt'
    sequences = FileManager.read_sequence_info(sequence_info_in, embeddings_tucker.keys())

    # The FunFams and sequence information can be directly extracted from FunFams and is included here to
    # allow using the code without access to the FunFams alignments

    fm = FileManager(ungapped_aln_path='path_to_save_data_for_each_superfamily_to',
                     dist_path_funfam='path_to_save_distances_to')
    # in the ungapped_aln_path directory, a sub-directory for each superfamily will be created.
    # In the directory for a superfamily, a director dist_path_funfam will be created to save the distances to
    # depending on the size of the FunFam, distance files can become rather large. Please make sure that enough
    # disk space is available

    # Calculate distances
    print("Calculate distances")
    dist_calculator = EmbeddingDistance(funfams, embeddings_tucker, sequences, fm)
    for f in funfams[superfamily]:
        if len(sequences[f]) > 1:
            dist_calculator.calc_distances_within_funfam(f, False)

    # Determine distance threshold
    print("Determine clustering cutoff")
    cutoff_determine = CutoffDetermination(superfamily, funfams[superfamily], None, fm)
    cutoff = cutoff_determine.determine_cutoff(0.5, 'sequence')
    print("Cutoff: {:.3f}".format(cutoff))

    # Define clustering outputs
    out_path = 'path_for_clustering_results'
    cluster_out_prefix = '{}clustering_results/'.format(out_path)
    outliers_out = '{}outliers.txt'.format(cluster_out_prefix)
    cluster_out_suffix = '_median_seq_tucker_protbert.txt'
    file_action = 'w'
    if os.path.exists(cluster_out_prefix):
        os.makedirs(cluster_out_prefix)

    # Perform clustering
    print("Perform clustering for each FunFam")
    for f in funfams[superfamily]:
        distance_file = '{}/{}/{}/{}_tucker_dist.npz'.format(fm.ungapped_aln_path, superfamily, fm.dist_path_funfam, f)
        if os.path.exists(distance_file):
            distances = dict(np.load(distance_file, mmap_mode='r'))['dist']
            if len(distances) > 0:
                clustering = DBSCANClustering(distances)
                cluster_results, outliers = clustering.calc_outliers_and_clustering(cutoff)

                # write output
                FileManager.write_outliers(outliers_out, f, outliers, file_action)
                cluster_results.write_clustering_results(f, cluster_out_prefix, cluster_out_suffix, file_action)
                file_action = 'a'  # append results to the previous file
