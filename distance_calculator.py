from file_manager import FileManager, Npy2Npz
from Bio import SeqIO
import torch
import numpy as np
from two_sample_util import pdist
from collections import defaultdict
import os


class EmbeddingDistance(object):
    """
    Calculate distances within and between FunFams using protein embeddings
    """

    def __init__(self, funfam_info, embeddings, sequences, fm):
        self.funfam_info = funfam_info
        self.embeddings = embeddings

        self.fm = fm
        if sequences is None:
            # read alignments from file if information is not yet available
            self.funfam_ids = self._get_sequences_in_funfam()
        else:
            self.funfam_ids = sequences
        self.family_sizes = self._get_relative_family_sizes()

    def calc_distances_within_funfam(self, f, return_results):
        """
        Calculate distances between all sequences within one FunFam
        :param f: FunFam for which distances should be calculated
        :param return_results: boolean to specify whether results should be directly written to disc or not
        (distance results can get rather big so it might make sense to write them directly to file if RAM is limited)
        :return: distances if not written to file, otherwise None
        """
        uniprot_single_seg = self.funfam_ids[f]
        results = []

        superfamily_folder = '{}/{}'.format(self.fm.ungapped_aln_path, f.split('-')[0])

        if not os.path.exists(superfamily_folder):
            os.makedirs(superfamily_folder)

        dist_folder = '{}/{}'.format(superfamily_folder, self.fm.dist_path_funfam)

        if not os.path.exists(dist_folder):
            os.makedirs(dist_folder)

        out_file = '{}/{}/{}/{}_tucker_dist'.format(self.fm.ungapped_aln_path, f.split('-')[0],
                                                    self.fm.dist_path_funfam, f)

        if not os.path.exists(out_file):
            if len(uniprot_single_seg) > 1:
                new_embeddings = self._get_embedding_subset(uniprot_single_seg)

                # calculate pairwise similarities for embeddings and write statistical information
                ids, raw_data = self._convert_to_tensor(new_embeddings)

                distances_tensor = pdist(raw_data, raw_data, 2)
                distances = np.array(distances_tensor.tolist(), dtype=np.float16)

                for i, uni1 in enumerate(ids):
                    ids_part = ids[i + 1:]
                    for i2, uni2 in enumerate(ids_part):
                        j = i2 + i + 1
                        dist = round(np.float16(distances[i][j]), 3)
                        key = '{}_{}'.format(uni1, uni2)
                        res = [key, dist]
                        results.append(res)
            if not return_results:
                self.save_distances(out_file, results)
                return None
        elif return_results:
            results = dict(np.load(out_file, mmap_mode='r'))['dist']

        return results

    def calc_distances_between_2_funfams(self, f1, f2, same_family, return_results):
        f1_name = f1[:4]

        sequences1 = self.funfam_ids[f1]
        sequences2 = self.funfam_ids[f2]

        results = []

        if same_family:
            f2_name = f2.split('-')[2]
            out_file = '{}/{}/{}/{}_{}_dist'.format(self.fm.ungapped_aln_path, f1.split('-'),
                                                    self.fm.dist_path_superfamily, f1_name, f2_name)
        else:
            f2_name = f2[:-4]
            out_file = '{}/{}/{}/{}_{}_dist'.format(self.fm.ungapped_aln_path, f1.split('-'),
                                                    self.fm.dist_path_outside, f1_name, f2_name)

        if not os.path.exists(out_file):
            if len(sequences1) > 0 and len(sequences2) > 0:
                embeddings1 = self._get_embedding_subset(sequences1)
                embeddings2 = self._get_embedding_subset(sequences2)

                ids1, raw_data1 = self._convert_to_tensor(embeddings1)
                ids2, raw_data2 = self._convert_to_tensor(embeddings2)

                # calculate pairwise distances
                distances_tensor = pdist(raw_data1, raw_data2, 2)
                distances = np.array(distances_tensor.tolist(), dtype=np.float16)

                results = self._generate_pair_results(ids1, ids2, distances)
                if not return_results:
                    self.save_distances(out_file, results)
                    return None
        elif return_results:
            results = dict(np.load(out_file, mmap_mode='r'))['dist']

        return results

    def calc_distances_between_funfams(self, f, rnd_factor, same_family, return_results, rand_funfams=None):
        """
        Calculate distances between FunFams where one FunFam is given and other FunFams are picked at random.
        :param f: FunFam for which calculations should be executed
        :param rnd_factor: How many random FunFams should be considered
        :param same_family: Should the chosen FunFams be from the same or different superfamilies?
        :param return_results: boolean to specify whether distances should be directly written to disc or not
        (distance results can get rather big so it might make sense to write them directly to file if RAM is limited)
        :param rand_funfams: list of randomly chosen FunFams to calculate distances to
        :return: distances if not written to file, otherwise pairs of FunFams used to calculate distances
        """

        uni_ids1 = self.funfam_ids[f]
        new_embeddings1 = self._get_embedding_subset(uni_ids1)
        f1_name = f[:-4]

        if rand_funfams is None:
            if same_family:
                rand_funfams = self._get_random_funfams(f, rnd_factor)
            else:
                rand_funfams = self._get_random_funfams_different_family(f, rnd_factor)

        all_results = defaultdict(list)
        pairs = []
        for f2 in rand_funfams:
            if same_family:
                f2_name = f2.split('-')[2][:-4]
            else:
                f2_name = f2[:-4]
            pairs.append('{}_{}'.format(f1_name, f2_name))
            key = '{}/{}'.format(f1_name, f2_name)

            if same_family:
                out_file = '{}/{}/{}/{}_{}_dist'.format(self.fm.ungapped_aln_path, f.split('-')[0],
                                                        self.fm.dist_path_superfamily, f1_name, f2_name)
            else:
                out_file = '{}/{}/{}/{}_{}_dist'.format(self.fm.ungapped_aln_path, f.split('-')[0],
                                                        self.fm.dist_path_outside, f1_name, f2_name)

            if not os.path.exists(out_file):
                uni_ids2 = self.funfam_ids[f2]
                new_embeddings2 = self._get_embedding_subset(uni_ids2)

                ids1, raw_data1 = self._convert_to_tensor(new_embeddings1)
                ids2, raw_data2 = self._convert_to_tensor(new_embeddings2)

                # calculate pairwise distances
                distances_tensor = pdist(raw_data1, raw_data2, 2)
                distances = np.array(distances_tensor.tolist(), dtype=np.float16)

                results = self._generate_pair_results(ids1, ids2, distances)
                if return_results:
                    all_results[key] = results
                else:
                    self.save_distances(out_file, results)
            elif return_results:
                results = dict(np.load(out_file, mmap_mode='r'))['dist']
                all_results[key] = results

        if return_results:
            return all_results
        else:
            return pairs

    @staticmethod
    def _generate_pair_results(ids1, ids2, distances):
        """
        Convert distance matrix in a 2d array with 1st column: seq. pair, 2nd column: distance
        :param ids1: column ids
        :param ids2: row ids
        :param distances: distance matrix
        :return: 2d array with 1st column: seq. pair, 2nd column: distance
        """
        results = []
        for x, uni1 in enumerate(ids1):
            for y, uni2 in enumerate(ids2):
                dist = round(np.float16(distances[x][y]), 3)
                key = '{}_{}'.format(uni1, uni2)
                res = [key, dist]
                results.append(res)

        return results

    def _get_random_funfams(self, f, rnd_factor):
        """
        Generate random FunFams from the same superfamily
        :param f: FunFam
        :param rnd_factor: number of random FunFams that should be chosen
        :return: list of random FunFams
        """

        cluster = f.split('-')[0]
        funfams = self.funfam_info[cluster]
        funfams_large = self._get_large_funfams(funfams)  # only use FunFams with >1 sequence with embeddings

        rand_f = [f]
        while f in rand_f:
            rand_f = np.random.choice(funfams_large, size=rnd_factor, replace=False)

        return rand_f

    def _get_random_funfams_different_family(self, f, rnd_factor):
        """
        Generate random FunFams from different superfamilies than given FunFam
        :param f: FunFam
        :param rnd_factor: number of random FunFams that should be chosen
        :return: list of random FunFams
        """
        family = f.split('-')[0]
        families = [family]
        while family in families:
            families = np.random.choice(list(self.family_sizes.keys()), size=rnd_factor, replace=True,
                                        p=list(self.family_sizes.values()))

        rand_funfams = []
        for fam in families:
            funfams = self.funfam_info[fam]
            funfams_large = self._get_large_funfams(funfams)  # only use FunFams with >1 sequence with embeddings

            rand_f = np.random.choice(funfams_large, size=1, replace=False)
            rand_funfams.append(rand_f)

        return rand_funfams

    def _get_sequences_in_funfam(self):
        """
        Read FunFam alignment without gaps and get sequences with embeddings
        :return: dictionary with key=FunFam, list=sequences with embeddings
        """

        funfam_ids = defaultdict(list)

        for cluster in self.funfam_info.keys():
            funfams = self.funfam_info[cluster]
            for f in funfams:
                ungapped_file = '{}/{}/seed_alignments/{}_nogaps'.format(self.fm.ungapped_aln_path, cluster, f)
                funfam_dict = SeqIO.to_dict(SeqIO.parse(ungapped_file, "fasta"))

                for i in funfam_dict.keys():
                    seq_id = funfam_dict[i].id

                    if seq_id in self.embeddings.keys():
                        funfam_ids[f].append(seq_id)

        return funfam_ids

    def _get_relative_family_sizes(self):
        """
        get relative size of superfamilies
        :return: dictionary with key=superfamily, value=relative size
        """
        family_sizes = dict()

        total_number = 0
        for c in self.funfam_info.keys():
            num = len(self.funfam_info[c])
            family_sizes[c] = num
            total_number += num

        for c in family_sizes.keys():
            family_sizes[c] = family_sizes[c] / total_number

        return family_sizes

    def _get_large_funfams(self, funfams):
        """
        Get FunFams with more than one sequence
        :param funfams: list of FunFams
        :return: list of FunFams with >1 sequence
        """
        funfams_large = []
        for f1 in funfams:
            num_uni = len(self.funfam_ids[f1])
            if num_uni > 1:
                funfams_large.append(f1)

        return funfams_large

    def _get_embedding_subset(self, ids):
        """
        For a set of ids, get the corresponding embeddings
        :param ids: ids to get embeddings for
        :return: embeddings
        """
        new_embeddings = dict()
        for i in ids:
            new_embeddings[i] = self.embeddings[i]
        return new_embeddings

    @staticmethod
    def _convert_to_tensor(embeddings):
        """
        Convert data to tensor
        :param embeddings:
        :return:
        """
        ids, raw_data = zip(*embeddings.items())
        raw_data = torch.tensor(raw_data, dtype=torch.float32).squeeze()
        if len(ids) == 1:
            raw_data = raw_data.unsqueeze(0)

        return ids, raw_data

    @staticmethod
    def save_distances(out_file, distances):
        """
        Save distances in compressed format
        :param out_file: file to which output should be written
        :param distances:
        :return:
        """
        np.savez_compressed(out_file, dist=distances)


def main():

    print("Read embeddings")
    path = 'data/'
    embeddings_in = '{}funfams_L1_tucker128.npy'.format(path)
    embedding_ids_in = '{}funfams_ids.txt'.format(path)
    embeddings = Npy2Npz.get_dataset_uncompressed(embeddings_in, embedding_ids_in)

    print("Read FunFam info")
    funfam_info_in = '{}cath_v4_3-funfams-summary.tsv'.format(path)
    funfam_info = FileManager.read_funfam_ids_with_family(funfam_info_in)

    print("Read sequence information")
    sequence_info_in = 'data/statistics_funfams_v4.3.txt'
    sequences = FileManager.read_sequence_info(sequence_info_in, embeddings.keys())

    fm = FileManager(ungapped_aln_path='funfam_data')
    dist_calculator = EmbeddingDistance(funfam_info, embeddings, sequences, fm)
    superfamilies_in = '{}funfam_binding_stats_test.txt'.format(path)
    superfamilies = FileManager.read_funfam_ids_with_family(superfamilies_in, 0, 1)

    print("Calculate distances")
    for s in superfamilies.keys():
        funfams = funfam_info[s]
        for f in funfams:
            if len(sequences[f]) > 1:
                print('{}\t{}'.format(s, f))
                dist_calculator.calc_distances_within_funfam(f, False)


if __name__ == '__main__':
    main()
