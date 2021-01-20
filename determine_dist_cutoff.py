import numpy as np
from collections import defaultdict
import sys
import os
from file_manager import FileManager


class CutoffDetermination(object):
    """
    Determine cutoff for a specific superfamily
    """

    def __init__(self, superfamily, funfam_ids, pair_ids, fm):
        self.superfamily = superfamily
        self.funfam_ids = funfam_ids
        self.pair_ids = pair_ids

        self.fm = fm

    def determine_cutoff(self, quantile, cutoff_type):
        """
        Determine a distance cutoff using a given quantile for a certain distance distribution
        :param quantile: quantile to use as cutoff
        :param cutoff_type: which cutoff to use (sequence: average distance of sequence to all others,
        family: distances between FunFams in the same superfamily)
        :return: cutoff
        """
        if cutoff_type == 'sequence':
            distances = self.get_mean_distances_sequence()
        elif cutoff_type == 'family':
            distances = self._get_distances_superfamily()
        else:
            sys.exit("{} is not a valid cutoff type. Choose 'sequence' or'family'".format(cutoff_type))

        if len(distances) > 0:
            if quantile == 'mean':
                cutoff = np.average(distances)
            elif quantile <= 1:  # quantile was given
                cutoff = np.quantile(distances, quantile)
            else:  # percentile was given
                cutoff = np.percentile(distances, quantile)
            cutoff = round(cutoff, 4)
        else:
            cutoff = None

        return cutoff

    def get_mean_distances_sequence(self, within_funfam=True, within_superfamily=True):
        """
        For each sequence in a FunFam, calculate the average distance to all other distances in this FunFam
        :return:
        """

        pairs_dict = defaultdict(list)
        for p in self.pair_ids:
            splitted_pair = p.split('_')
            p1 = splitted_pair[0]
            p2 = splitted_pair[1]

            pairs_dict[p1].append(p2)

        mean_sequence_distances = []
        for f in self.funfam_ids:
            f = f[:-4]
            distances = [[]]
            if within_funfam:
                distance_file = '{}/{}/{}/{}_tucker_dist.npz'.format(self.fm.ungapped_aln_path, self.superfamily,
                                                                     self.fm.dist_path_funfam, f)
                if os.path.exists(distance_file):
                    # only analyse Funfams for which distances where calculated
                    distances = dict(np.load(distance_file, mmap_mode='r'))['dist']

            elif not within_funfam and within_superfamily:
                for p in pairs_dict[f]:
                    distance_file = '{}/{}/{}/{}_{}_dist.npz'.format(self.fm.ungapped_aln_path, self.superfamily,
                                                                     self.fm.dist_path_superfamily, f, p)
                    if os.path.exists(distance_file):
                        tmp_distances = dict(np.load(distance_file, mmap_mode='r'))['dist']
                        if np.size(distances) > 0:
                            distances = np.concatenate((distances, tmp_distances), axis=0)
                        else:
                            distances = tmp_distances
            else:
                sys.exit('Mode currently not implemented')

            if np.size(distances) > 0:
                sequence_statistics = self._calc_sequence_statistics(distances)
                for s in sequence_statistics.keys():
                    mean_sequence_distances.append(sequence_statistics[s]['mean'])

        return mean_sequence_distances

    def _get_distances_superfamily(self):
        """
        Get distances between sequences in different superfamilies
        :return:
        """
        all_distances = []
        for f in self.pair_ids:
            distance_file = '{}/{}/{}/{}_tucker_dist.npz'.format(self.fm.ungapped_aln_path, self.superfamily,
                                                                 self.fm.dist_path_superfamily, f)
            distances = dict(np.load(distance_file, mmap_mode='r'))['dist']
            distance_values = np.array(distances[:, 1], dtype=np.float16)
            all_distances.append(distance_values)

        return all_distances

    @staticmethod
    def _calc_sequence_statistics(distances):
        """
        For each sequence in a FunFam, calculate min, q1, median, mean, q3, max, and range to all other sequences
        in the FunFam
        :param distances: pairwise distances of a FunFam
        :return: calculated statistics
        """
        sequence_statistics = defaultdict(dict)
        sequence_distances = defaultdict(list)

        for d in distances:
            pair = d[0]
            dist = float(d[1])
            split_pair = pair.split('_')

            sequence_distances[split_pair[0]].append(dist)
            sequence_distances[split_pair[1]].append(dist)

        for k in sequence_distances.keys():
            distance_values = sequence_distances[k]
            min_dist = round(float(np.amin(distance_values)), 3)
            max_dist = round(float(np.amax(distance_values)), 3)
            mean_dist = round(float(np.mean(distance_values)), 3)
            q1 = round(np.percentile(distance_values, 25), 3)
            median = round(np.percentile(distance_values, 50), 3)
            q3 = round(np.percentile(distance_values, 75), 3)
            val_range = round(max_dist - min_dist, 3)

            sequence_statistics[k] = {'min': min_dist, 'max': max_dist, 'mean': mean_dist, 'q1': q1, 'median': median,
                                      'q3': q3, 'val_range': val_range}

        return sequence_statistics


def main():
    path = 'data/'
    print("Read FunFam info")
    funfam_info_in = '{}cath_v4_3-funfams-summary.tsv'.format(path)
    funfam_info = FileManager.read_funfam_ids_with_family(funfam_info_in)
    superfamilies_in = '{}funfam_binding_stats_test.txt'.format(path)
    superfamilies = FileManager.read_funfam_ids_with_family(superfamilies_in, 0, 1)

    fm = FileManager(ungapped_aln_path='funfam_data')
    cutoffs = dict()
    print("Determine distance cutoff")
    for i, s in enumerate(list(superfamilies.keys())):
        print('{}\t{}'.format(i, s))
        funfams = funfam_info[s]
        cutoff_determine = CutoffDetermination(s, funfams, None, fm)
        cutoff = cutoff_determine.determine_cutoff(0.5, 'sequence')
        cutoffs[s] = cutoff

    out_file = '{}distance_cutoffs_median.txt'.format(path)
    FileManager.write_dictionary(out_file, cutoffs)


if __name__ == '__main__':
    main()
