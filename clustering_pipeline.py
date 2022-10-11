import numpy as np
import math
from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN
from collections import defaultdict, Counter
from file_manager import FileManager
import os
import sys


class DBSCANClustering(object):
    """
    Class to perform DBSCAN clustering
    """

    def __init__(self, distances, neighborhood_size=5, digits=1, num_seq=0, matrix=False, ids=None):

        if not matrix:
            # construct distance matrix
            if num_seq == 0:
                self.num_seq = int((1 + math.sqrt(1 + 8 * len(distances[:, 0]))) / 2)
            else:
                self.num_seq = num_seq
            self.distance_matrix = np.zeros((self.num_seq, self.num_seq))
            self.seq_order = []

            # print(self.num_seq)
            # print(np.shape(self.distance_matrix))

            for d in distances:
                pair = d[0]
                dist = float(d[1])
                split_pair = pair.split('!!')
                a = split_pair[0]
                b = split_pair[1]

                if a not in self.seq_order:
                    self.seq_order.append(a)
                if b not in self.seq_order:
                    self.seq_order.append(b)

                ind_a = self.seq_order.index(a)
                ind_b = self.seq_order.index(b)
                self.distance_matrix[ind_a][ind_b] = dist
                self.distance_matrix[ind_b][ind_a] = dist

        else:
            self.distance_matrix = distances
            self.seq_order = ids
            self.num_seq = len(ids)

        self.max_dist = round(float(np.amax(self.distance_matrix)), 3)
        self.epsilons = np.arange(0.1, 1.1, 0.1).tolist()
        # self.neighborhood_size = max(neighborhood_size, math.ceil(0.01 * len(self.seq_order)))
        self.neighborhood_size = neighborhood_size
        self.digits = digits

    def calc_outliers_and_clustering(self, dist_cutoff, method='dbscan'):
        """
        Get clustering and outlier probabilites using a certain distance cutoff
        :param dist_cutoff: cutoff to use for clustering
        :param method: to specify cluster methods, default: dbscan
        :return: clusters
        """

        if method == 'dbscan':
            cluster_result, outliers = self._calc_outliers_and_clustering_dbscan(dist_cutoff)
        elif method == 'optics':
            cluster_result, outliers = self._calc_outliers_and_clustering_optics()
        elif method == 'hdbscan':
            cluster_result, outliers = self._calc_outliers_and_clustering_hdbscan()
        else:
            sys.exit("Not a valid clustering")

        return cluster_result, outliers

    def _calc_outliers_and_clustering_dbscan(self, dist_cutoff):

        outliers = dict()
        outliers_clustering = {-1: []}
        sequence_clusters = {0: self.seq_order}

        size_list = '[{}]'.format(self.num_seq)
        num_clusters = 1
        num_outliers = 0

        for e in self.epsilons:
            if round(e, self.digits) <= self.max_dist:
                clustering = DBSCAN(eps=e, min_samples=self.neighborhood_size,
                                    metric="precomputed").fit(self.distance_matrix)
                # scale distance at which a sequence gets classified as an outlier by the maximum distance to represent
                # a probability
                proba = round(e / self.max_dist, 5)
                for index, label in enumerate(clustering.labels_):
                    if label == -1:  # outlier
                        seq = self.seq_order[index]
                        if (seq in outliers.keys() and outliers[seq] < proba) or \
                                (seq not in outliers.keys()):
                            outliers[seq] = proba

                # save clustering if clustering is at right threshold
                if math.isclose(round(e, self.digits), round(dist_cutoff, self.digits)):
                    print("Perform clustering")
                    c = ClusterAnalysis(clustering.labels_)
                    sequence_clusters, outliers_clustering = c.get_sequence_clusters(self.seq_order)
                    size_list, num_clusters, num_outliers = c.get_num_clusters_outliers()

                    print('{}\t{}'.format(num_clusters, num_outliers))

        cluster_result = ClusteringResults(sequence_clusters, outliers_clustering, num_clusters,
                                           num_outliers, size_list)

        return cluster_result, outliers

    def _calc_outliers_and_clustering_hdbscan(self):

        outliers = dict()

        print("Perform clustering")
        clustering = HDBSCAN(metric='precomputed')
        clustering.fit(self.distance_matrix)
        c = ClusterAnalysis(clustering.labels_)
        sequence_clusters, outliers_clustering = c.get_sequence_clusters(self.seq_order)
        size_list, num_clusters, num_outliers = c.get_num_clusters_outliers()

        cluster_result = ClusteringResults(sequence_clusters, outliers_clustering, num_clusters, num_outliers,
                                           size_list)

        return cluster_result, outliers

    def _calc_outliers_and_clustering_optics(self):

        outliers = dict()

        clustering = OPTICS(min_samples=math.ceil(0.1*len(self.seq_order))).fit(self.distance_matrix)
        print("Perform clustering")
        c = ClusterAnalysis(clustering.labels_)
        sequence_clusters, outliers_clustering = c.get_sequence_clusters(self.seq_order)
        size_list, num_clusters, num_outliers = c.get_num_clusters_outliers()

        # print(num_clusters)
        # print(num_outliers)

        cluster_result = ClusteringResults(sequence_clusters, outliers_clustering, num_clusters, num_outliers,
                                           size_list)

        return cluster_result, outliers

    def calc_clustering(self, dist_cutoff, method='dbscan'):
        """
        Get DBSCAN clustering at a certain distance cutoff
        :param dist_cutoff:
        :param method:
        :return:
        """
        outliers_clustering = {-1: []}
        sequence_clusters = {0: self.seq_order}
        size_list = '[{}]'.format(self.num_seq)
        num_clusters = 1
        num_outliers = 0

        if method == 'dbscan':
            if dist_cutoff <= self.max_dist:
                clustering = DBSCAN(eps=dist_cutoff, min_samples=self.neighborhood_size,
                                    metric="precomputed").fit(self.distance_matrix)
                c = ClusterAnalysis(clustering.labels_)
                sequence_clusters, outliers_clustering = c.get_sequence_clusters(self.seq_order)
                size_list, num_clusters, num_outliers = c.get_num_clusters_outliers()
        elif method == 'hdbscan':
            clustering = HDBSCAN(metric="precomputed")
            clustering.fit(self.distance_matrix)
            c = ClusterAnalysis(clustering.labels_)

            sequence_clusters, outliers_clustering = c.get_sequence_clusters(self.seq_order)
            size_list, num_clusters, num_outliers = c.get_num_clusters_outliers()
        else:
            sys.exit('Not a valid clustering method')

        cluster_result = ClusteringResults(sequence_clusters, outliers_clustering, num_clusters, num_outliers,
                                           size_list)

        return cluster_result


class ClusterAnalysis(object):
    """
    Class to analyse generated clustering
    """

    def __init__(self, clusters):
        self.clusters = clusters
        self.size_clusters = Counter(self.clusters)

    def get_sequence_clusters(self, seq_order):
        """
        Get mapping which sequences were assigned to which cluster
        :param seq_order: sequences that were clustered
        :return: dictionary with key=cluster number, value=sequences in this cluster
        """
        clusters = defaultdict(list)
        outliers = defaultdict(list)

        for c in self.size_clusters.keys():
            indices = [i for i, x in enumerate(self.clusters) if x == c]
            sequences = [seq_order[i] for i in indices]
            if c == -1:
                outliers[-1] = sequences
            else:
                clusters[c] = sequences

        return clusters, outliers

    def get_cluster_sizes(self):
        """
        Get sizes of each cluster and format them as string with the numbers occurring in the right order
        (size at position i = size of cluster i)
        :return: formatted string of cluster sizes
        """
        size_list = ''
        for c in sorted(list(self.size_clusters.keys())):
            if c != -1:
                size_list += ',{}'.format(self.size_clusters[c])
        size_list = size_list[1:]
        if size_list == '':
            size_list = '0'
        size_list = '[{}]'.format(size_list)

        return size_list

    def get_num_clusters_outliers(self):
        """
        Get the list of sizes, the number of clusters and the number of outliers for this clustering
        Outliers are not counted as clusters
        :return: size list, number of clusters, number of outliers
        """
        size_list = self.get_cluster_sizes()
        num_clusters = len(self.size_clusters.keys())
        num_outliers = 0

        if -1 in self.size_clusters.keys():
            num_outliers = self.size_clusters[-1]
            num_clusters += -1

        return size_list, num_clusters, num_outliers


class ClusteringResults(object):
    """
    Wrapper for interesting results of clustering
    """

    def __init__(self, clusters, outliers, num_clusters, num_outliers, size_list):
        self.clusters = clusters
        self.outliers = outliers
        self.num_clusters = num_clusters
        self.num_outliers = num_outliers
        self.size_list = size_list

        self.sizes = self._calc_relative_sizes()

    def write_clustering_results(self, funfam, out_prefix, out_suffix, file_action):
        # define output files
        cluster_summary_out = '{}cluster_summary{}'.format(out_prefix, out_suffix)
        clusters_out = '{}clusters{}'.format(out_prefix, out_suffix)
        cluster_outliers_out = '{}outliers{}'.format(out_prefix, out_suffix)
        cluster_sizes_out = '{}cluster_relative_sizes{}'.format(out_prefix, out_suffix)

        # convert clusters and outliers to string
        cluster_str = ClusteringResults._clusters_to_string(self.clusters)
        outliers_str = ClusteringResults._clusters_to_string(self.outliers)

        # write output
        FileManager.write_cluster_summary(cluster_summary_out, funfam, self.num_clusters, self.num_outliers,
                                          self.size_list, file_action)
        FileManager.write_clusters(clusters_out, funfam, cluster_str, file_action)
        FileManager.write_clusters(cluster_outliers_out, funfam, outliers_str, file_action)
        FileManager.write_relative_sizes(cluster_sizes_out, funfam, self.sizes, file_action)

    def get_consensus_sequences(self, sequences, binding_residues):
        all_sequences = defaultdict(set)
        cluster_sequences = defaultdict(set)

        all_seqs = self._get_all_sequences()

        # check whether there exist cluster with sequences with binding annotations
        for c in self.clusters.keys():
            cluster_seqs = self._get_sequences_in_cluster(self.clusters[c])
            for s in self.clusters[c]:
                seq = s.split('/')[0].split('.')[0]
                if seq in sequences:
                    binding_res = binding_residues[seq]
                    seq_range = s.split('/')[1].split('-')
                    start = int(seq_range[0])
                    end = int(seq_range[1])

                    for r in range(start, end + 1):
                        if r in binding_res:
                            # seq is a query sequence with binding annotations
                            # add all sequences for consensus prediction for FunFam
                            all_sequences[s] = all_seqs
                            # add sequences in cluster for consensus prediction of cluster
                            cluster_sequences[s] = cluster_seqs

        # check whether there exist outliers with binding annotations
        for o in self.outliers.keys():
            for s in self.outliers[o]:
                seq = s.split('/')[0].split('.')[0]
                if seq in sequences:
                    binding_res = binding_residues[seq]
                    seq_range = s.split('/')[1].split('-')
                    start = int(seq_range[0])
                    end = int(seq_range[1])

                    for r in range(start, end + 1):
                        if r in binding_res:
                            # seq is a query sequence with binding annotations
                            # add all sequences for consensus prediction for FunFam
                            all_sequences[s] = all_seqs
                            # add only this sequence for consensus prediction of cluster (cluster of 1 sequence)
                            seq_set = set()
                            seq_set.add(s)
                            cluster_sequences[s] = seq_set

        return all_sequences, cluster_sequences

    @staticmethod
    def _clusters_to_string(clusters):
        cluster_str = ''
        cluster_keys = sorted(list(clusters.keys()))
        for c in cluster_keys:
            cluster_ids = clusters[c]
            c_str = ''
            for i in cluster_ids:
                c_str += ',{}'.format(i)
            c_str = c_str[1:]
            c_str = '[{}]'.format(c_str)
            cluster_str += ';{}'.format(c_str)
        cluster_str = cluster_str[1:]
        return cluster_str

    def _calc_relative_sizes(self):
        sizes = dict()
        size = self.size_list[1:-1].split(',')
        size = np.array(size, dtype=np.int)
        sum_size = np.sum(size)

        for i, s in enumerate(size):
            rel_size = round(s / sum_size, 3)
            sizes[i] = rel_size

        return sizes

    def _get_all_sequences(self):
        all_sequences = set()
        for c in self.clusters.keys():
            sequences = self._get_sequences_in_cluster(self.clusters[c])
            all_sequences.update(sequences)

        if -1 in self.outliers.keys():
            sequences = self._get_sequences_in_cluster(self.outliers[-1])
            all_sequences.update(sequences)

        return all_sequences

    @staticmethod
    def _get_sequences_in_cluster(cluster):
        sequences = set()
        for s in cluster:
            # seq = s.split('/')[0].split('.')[0]
            sequences.add(s)

        return sequences


def main():
    print("Read data")
    path = 'data/'
    superfamilies_in = '{}funfam_binding_stats_test.txt'.format(path)
    superfamilies = FileManager.read_funfam_ids_with_family(superfamilies_in, 0, 1)
    cutoffs_in = '{}distance_cutoffs_median.txt'.format(path)
    cutoffs = FileManager.read_dictionary(cutoffs_in, 'float')

    fm = FileManager(ungapped_aln_path='funfam_data')
    file_action = 'w'

    print("Define output files")
    cluster_out_prefix = 'clustering_results/binding_test/'
    outliers_out = '{}outliers_binding_test.txt'.format(cluster_out_prefix)
    cluster_out_suffix = '_median_seq_tucker.txt'

    print("Run DBSCAN")
    for s in superfamilies:
        funfams = superfamilies[s]
        dist_cutoff = cutoffs[s]
        for f in funfams:
            distance_file = '{}/{}/{}/{}_tucker_dist.npz'.format(fm.ungapped_aln_path, s, fm.dist_path_funfam, f)
            if os.path.exists(distance_file):
                distances = dict(np.load(distance_file, mmap_mode='r'))['dist']
                if len(distances) > 0:
                    clustering = DBSCANClustering(distances)
                    print('{}\t{}\t{}\t{}'.format(s, f, clustering.max_dist, dist_cutoff))
                    cluster_results, outliers = clustering.calc_outliers_and_clustering(dist_cutoff)

                    # write output
                    FileManager.write_outliers(outliers_out, f, outliers, file_action)
                    cluster_results.write_clustering_results(f, cluster_out_prefix, cluster_out_suffix, file_action)
                    file_action = 'a'


if __name__ == '__main__':
    main()
