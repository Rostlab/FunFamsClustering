from collections import defaultdict
import random
import numpy as np


class FileManager(object):

    def __init__(self, ungapped_aln_path='path_to_save_data_to',
                 dist_path_superfamily='family_dist_tucker', dist_path_funfam='funfam_dist_tucker',
                 dist_path_outside='outside_dist_tucker'):
        self._ungapped_aln_path = ungapped_aln_path
        self._dist_path_superfamily = dist_path_superfamily
        self._dist_path_funfam = dist_path_funfam
        self._dist_path_outside = dist_path_outside

    @property
    def ungapped_aln_path(self):
        return self._ungapped_aln_path

    @property
    def dist_path_superfamily(self):
        return self._dist_path_superfamily

    @property
    def dist_path_funfam(self):
        return self._dist_path_funfam

    @property
    def dist_path_outside(self):
        return self._dist_path_outside

    @staticmethod
    def read_ids(file_in):
        """
        Read ID list (take first value in each row)
        :param file_in: Input file, tab-separated, 1st element is ID
        :return:
        """
        ids = []
        with open(file_in) as read_in:
            for line in read_in:
                ids.append(line.strip().split()[0])

        return ids

    @staticmethod
    def read_ids_dict(file_in):

        ids = dict()
        with open(file_in) as read_in:
            for line in read_in:
                identifier = line.strip().split()[0]
                ids[identifier] = 1

        return ids

    @staticmethod
    def read_funfam_ids_with_family(file_in, family=-1, funfam=0):
        """
        Create dict of superfamily -> list of FunFams
        :param file_in: Input file, tab-separated, 1st element is FunFam ID
        :param family: Which column contains the superfamily. If -1 -> get superfamily from FunFam ID
        :param funfam: Which column contains the FunFam
        :return:
        """
        info = defaultdict(list)
        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_line = line.strip().split()
                funfam_id = splitted_line[funfam]

                if family == -1:
                    cluster = funfam_id.split('-')[0]
                else:
                    cluster = splitted_line[family]

                info[cluster].append(funfam_id)

        return info

    @staticmethod
    def read_sequence_info(file_in, embeddings):
        """
        Read sequences for each FunFam from FunFam stats format & only consider sequences with embeddings
        :param file_in: FunFam stats file of format FunFamID\tSeq.Identifier\tSeq.Segment
        :param embeddings: protein ids with embeddings
        :return:
        """
        sequences = defaultdict(list)
        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_line = line.strip().split()

                funfam_id = splitted_line[0]
                identifier = splitted_line[1].split('.')[0]
                segments = splitted_line[2]

                uni_id = '{}/{}'.format(identifier, segments)

                if uni_id in embeddings:
                    sequences[funfam_id].append(uni_id)

        return sequences

    @staticmethod
    def read_funfam_sequences(file_in):
        """
        Get FunFams and set of sequences in these FunFams
        :param file_in: Input file
        :return:
        """
        funfam_info = defaultdict(set)
        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_line = line.strip().split()

                f = splitted_line[0]
                uni_id = splitted_line[1].split('.')[0]
                segment = splitted_line[2]
                uni_seg = '{}/{}'.format(uni_id, segment)

                funfam_info[f].add(uni_seg)

        return funfam_info

    @staticmethod
    def read_funfam_stats(file_in):
        funfam_info = defaultdict(defaultdict)
        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_line = line.strip().split()

                f = splitted_line[0]
                uni_id = splitted_line[1]
                segment = splitted_line[2]
                length = splitted_line[3]
                annotated = splitted_line[4]

                funfam_info[f][uni_id] = {'Segments': segment, 'Length': length, 'Ligand': annotated}

        return funfam_info

    @staticmethod
    def read_clusters_clust_id(file_in, start_id):
        """
        Read clusters for one FunFam and assign each cluster a unique clustering id
        :param file_in: Input file, format: FunFamID\tCluster1;Cluster2;...
                Sequences in one cluster are separated by ','
        :param start_id: Start ID for clustering
        :return:
        """
        cluster_info = defaultdict(defaultdict)

        with open(file_in) as read_in:
            for line in read_in:
                splitted_line = line.strip().split()
                funfam_id = splitted_line[0]
                if len(splitted_line) > 1:
                    clusters = splitted_line[1].split(';')
                    clust_id = start_id
                    for c in clusters:
                        c = c[1:-1]
                        if c != '':  # empty list
                            sequences = set(c.split(','))
                            cluster_info[funfam_id][clust_id] = sequences
                            clust_id += 1
                        else:
                            cluster_info[funfam_id] = defaultdict(list)
                else:
                    cluster_info[funfam_id] = defaultdict(list)

        return cluster_info

    @staticmethod
    def read_cluster_summary(file_in):
        """
        Read cluster summary (number of clusters, number of outliers, sizes) from tab-separated file
        :param file_in: Input file
        :return:
        """
        summary = defaultdict(dict)
        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_line = line.strip().split()
                f = splitted_line[0]
                num_clust = int(splitted_line[1])
                num_out = int(splitted_line[3])
                size_list = splitted_line[2]

                summary[f] = {'num_clust': num_clust, 'num_out': num_out, 'size_list': size_list}

        return summary

    @staticmethod
    def read_clusters_sizes_by_funfam(file_in):
        """
        Read cluster sizes
        :param file_in: Input file
        :return:
        """
        clusters = defaultdict(dict)
        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_line = line.strip().split()
                funfam = splitted_line[0]
                num_outliers = int(splitted_line[3])
                cluster_sizes = splitted_line[2][1:-1].split(',')

                if num_outliers > 0:
                    clusters[funfam][-1] = num_outliers

                for i, c in enumerate(cluster_sizes):
                    clusters[funfam][i] = int(c)

        return clusters

    @staticmethod
    def read_outliers(file_in):
        """
        Read outlier sequences
        :param file_in: Input file, tab-separated. Format: Superfamily\tFunFam\tSequence\tProba
        :return:
        """
        outliers = defaultdict(dict)

        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_line = line.strip().split()
                funfam = splitted_line[1]
                sequence = splitted_line[2]
                proba = float(splitted_line[3])

                outliers[funfam][sequence] = proba

        return outliers

    @staticmethod
    def read_ec_per_sequence(file_in, level=4):
        """
        Read EC numbers per FunFam up to a certain level
        :param file_in: Input
        :param level: Level until which ECs should be read (how specific do annotations have to be to be considered?)
        :return:
        """
        ec_per_seq = defaultdict(set)
        with open(file_in) as read_in:
            for line in read_in:
                splitted_line = line.strip().split()
                identifier = splitted_line[0]
                ec = splitted_line[1]

                unknown_numbers = ec.count('-')
                level_of_ec = 4 - unknown_numbers

                if level_of_ec >= level:
                    ec_per_seq[identifier].add(ec)

        return ec_per_seq

    @staticmethod
    def read_ec_analysis(file_in):
        """
        Read EC information from tab-separated file (Size, Num.EC, Num.Anno)
        :param file_in: tab-separated input file
        :return:
        """
        ec_analysis = defaultdict(dict)
        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_id = line.strip().split()
                funfam_id = splitted_id[1]
                size = int(splitted_id[2])
                num_ec = int(splitted_id[3])
                num_anno = int(splitted_id[4])

                ec_analysis[funfam_id] = {'size': size, 'num_ec': num_ec, 'num_anno': num_anno}

        return ec_analysis

    @staticmethod
    def read_binding_residues(file_in):
        """
        Read binding residues
        :param file_in: Input file, format: Prot.ID\tRes1,Res2,...
        :return:
        """
        binding_residues = defaultdict(list)
        with open(file_in) as read_in:
            for line in read_in:
                splitted_line = line.strip().split()
                identifier = splitted_line[0]
                residues = splitted_line[1].split(',')
                residues = [int(r) for r in residues]
                binding_residues[identifier] = residues

        return binding_residues

    @staticmethod
    def read_hmm_results(file_in, funfam):
        """
        Read E-values to HMMs
        :param file_in: Input file
        :param funfam: FunFam to consider
        :return:
        """
        sequences = dict()

        with open(file_in) as read_in:
            for line in read_in:
                if '#' not in line:
                    splitted_line = line.strip().split()
                    f = splitted_line[0]
                    if f == funfam:
                        seq_id = splitted_line[2]
                        e_value = float(splitted_line[4])

                        sequences[seq_id] = e_value

        return sequences

    @staticmethod
    def read_dictionary(file_in, convert=None):
        """
        Read 2-column tab-separated file as dictionary
        :param file_in: Input file
        :param convert: None: No conversion, 'float': Convert value to float, 'int': Convert value to int
        :return:
        """
        dictionary = dict()
        with open(file_in) as read_in:
            for line in read_in:
                splitted_line = line.strip().split()
                identifier = splitted_line[0]
                value = splitted_line[1]
                if convert == 'float':
                    value = float(value)
                elif convert == 'int':
                    value = int(value)

                dictionary[identifier] = value

        return dictionary

    @staticmethod
    def read_ligand_annotations(file_in):
        """
        Read ligand information and bound residues
        :param file_in: Input file
        :return: Dictionary with Key=Uniprot ID, Value=Dictionary with Key=Ligand, Value=Residues
        """

        ligand_annotations = defaultdict(defaultdict)

        with open(file_in) as read_in:
            next(read_in)
            for line in read_in:
                splitted_line = line.strip().split()
                prot_id = splitted_line[0]
                ligand = splitted_line[1]
                residues = set(splitted_line[2].split(','))

                ligand_annotations[prot_id][ligand] = residues

        return ligand_annotations
    #
    # @staticmethod
    # def read_ec_info(file_in):
    #     ec_info = defaultdict(set)
    #     with open(file_in) as read_in:
    #         for line in read_in:
    #             splitted_line = line.strip().split()
    #             identifier_splitted = splitted_line[0].split('.')
    #             identifier = '.'.join(identifier_splitted[0:4])
    #             if 'has no EC terms' in line:
    #                 ec_info[identifier] = set()
    #             else:
    #                 for i in range(3, len(splitted_line)):
    #                     if i % 2 == 1:
    #                         ec_num = splitted_line[i]
    #                         ec_info[identifier].add(ec_num)
    #
    #     return ec_info
    #
    #
    # @staticmethod
    # def write_ec_analysis(file_out, content):
    #     with open(file_out, 'w') as out:
    #         out.write('ID\tNum.Anno\tNum.Clusters\t0Annos\t1Anno\t2Annos\t3Annos\t4Annos\tOutliers\n')
    #         for k in content.keys():
    #             out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(k, content[k]['anno'],
    #                                                                     content[k]['cluster_num'], content[k]['0_anno'],
    #                                                                     content[k]['1_anno'], content[k]['2_anno'],
    #                                                                     content[k]['3_anno'], content[k]['4_anno'],
    #                                                                     content[k]['num_outlier']))
    #
    # @staticmethod
    # def write_distance_thresholds(file_out, content, file_action):
    #     with open(file_out, file_action) as out:
    #         for c in content.keys():
    #             out.write('{}\t{}\n'.format(c, content[c]))
    #
    # @staticmethod
    # def write_outlier_purity(probabilities, file_out):
    #     with open(file_out, 'w') as out:
    #         out.write('Cluster\tFunFam\tProba\tFract.Outliers\n')
    #         for f in probabilities.keys():
    #             cluster = f.split('-')[0]
    #             out.write('{}\t{}\t{}\t{}\n'.format(cluster, f, probabilities[f]['proba'], probabilities[f]['fract']))
    #
    # @staticmethod
    # def write_ec_info(file_out, info):
    #     with open(file_out, 'w') as out:
    #         out.write('Cluster\tID\tSize\tNum.EC\tNum.Annotations\n')
    #         for k in info.keys():
    #             c = k.split('-')[0]
    #             out.write('{}\t{}\t{}\t{}\t{}\n'.format(c, k, info[k]['size'], info[k]['ec'], info[k]['anno']))
    #
    # @staticmethod
    # def write_consensus_summary(file_out, content):
    #     with open(file_out, 'w') as out:
    #         out.write('FunFam\tNum.Query\tNum.Sequences\n')
    #         for k in content.keys():
    #             out.write('{}\t{}\t{}\n'.format(k, content[k]['num_query'], content[k]['num_seq']))
    #
    #
    # @staticmethod
    # def write_set(file_out, content, with_regions=True):
    #     """
    #     Write list of
    #     :param file_out:
    #     :param content:
    #     :param with_regions:
    #     :return:
    #     """
    #     with open(file_out, 'w') as out:
    #         for s in content:
    #             if not with_regions:
    #                 seq = s.split('/')[0]
    #             else:
    #                 seq = s
    #             out.write('{}\n'.format(seq))
    #
    # @staticmethod
    # def write_set_no_regions(file_out, content):
    #     with open(file_out, 'w') as out:
    #         for s in content:
    #             seq = s.split('/')[0]
    #             out.write('{}\n'.format(seq))
    #
    # @staticmethod
    # def write_defaultdict(file_out, content):
    #     with open(file_out, 'w') as out:
    #         out.write('FunFam\tNum.Cluster\tNum.Impure\n')
    #         for k in content.keys():
    #             out.write('{}\t{}\t{}\n'.format(k, content[k]['cluster_num'], content[k]['cluster_impure']))
    #
    # @staticmethod
    # def write_ec_summary(content, file_out):
    #     with open(file_out, 'w') as out:
    #         out.write('Superfamily,0_EC,1_EC,2_EC,3_EC,4_or_more_EC\n')
    #         for k in content.keys():
    #             out.write('{},{},{},{},{},{}\n'.format(k, content[k][0], content[k][1], content[k][2], content[k][3],
    #                                                    content[k][4]))

    @staticmethod
    def write_cluster_summary(file_out, funfam, num_clusters, num_outliers, size_list, file_action):
        """
        Write cluster summary for given FunFam
        :param file_out: Output file
        :param funfam: FunFam ID
        :param num_clusters: Number of clusters
        :param num_outliers: Number of outliers
        :param size_list: List of cluster sizes
        :param file_action: whether to overwrite or append to the existing file (w|a)
        :return:
        """
        with open(file_out, file_action) as out:
            if file_action == 'w':
                out.write('Funfam\tNumCluster\tSizeCluster\tNumOutliers\n')
            out.write('{}\t{}\t{}\t{}\n'.format(funfam, num_clusters, size_list, num_outliers))

    @staticmethod
    def write_clusters(file_out, funfam, cluster_str, file_action):
        """
        Write clusters for given FunFam
        :param file_out: Output file
        :param funfam: FunFam ID
        :param cluster_str: List of clusters, format: sequences in one cluster are separated by ',';
                            clusters are separated by ';'
        :param file_action: whether to overwrite or append to the existing file (w|a)
        :return:
        """
        with open(file_out, file_action) as out:
            out.write('{}\t{}\n'.format(funfam, cluster_str))

    @staticmethod
    def write_relative_sizes(file_out, funfam, sizes, file_action):
        """
        Write relative size for each cluster for given FunFam
        :param file_out: Output file
        :param funfam: FunFam ID
        :param sizes: List of relative cluster sizes (sums to 1)
        :param file_action: whether to overwrite or append to the existing file (w|a)
        :return:
        """
        c = funfam.split('-')[0]
        with open(file_out, file_action) as out:
            if file_action == 'w':
                out.write('Cluster\tFunFam\tRel.Size\n')
            for s in sizes.keys():
                new_id = '{}.{}'.format(funfam, s)
                out.write('{}\t{}\t{}\n'.format(c, new_id, sizes[s]))

    @staticmethod
    def write_outliers(file_out, funfam, outliers, file_action):
        """
        Write outlier sequence for given FunFam with outlier probability
        :param file_out: Output file
        :param funfam: FunFam ID
        :param outliers: Outlier sequences + probabilities
        :param file_action: whether to overwrite or append to the existing file (w|a)
        :return:
        """
        family = funfam.split('-')[0]
        with open(file_out, file_action) as out:
            if file_action == 'w':
                out.write('Cluster\tFunFam\tSequence\tProbability\n')
            for o in outliers.keys():
                out.write('{}\t{}\t{}\t{}\n'.format(family, funfam, o, outliers[o]))

    @staticmethod
    def write_consensus_info(file_out, content):
        """
        Write files that can be used for consensus prediction. For each sequence to use, one line is written
        :param file_out: Output file in format ID\tQuery\tSequence
        :param content:
        :return:
        """
        with open(file_out, 'w') as out:
            for k in content.keys():
                for q in content[k].keys():
                    for s in content[k][q]:
                        # seq = s.split('/')[0]
                        out.write('{}\t{}\t{}\n'.format(k, q, s))

    @staticmethod
    def write_dictionary(file_out, content, header=None, sort=False, file_action='w'):
        """
        Write dictionary (with header) to file
        :param file_out: Output file
        :param content: Dictionary to write
        :param header: Optional (header to write as 1st line)
        :param sort: Should keys be sorted?
        :param file_action: Should content be appended or written to a new file
        :return:
        """

        keys = list(content.keys())
        if sort:
            keys.sort()

        with open(file_out, file_action) as out:
            if header is not None:
                out.write(header)
            for k in keys:
                out.write('{}\t{}\n'.format(k, content[k]))

    @staticmethod
    def write_ids(content, file_out, split_id=None):
        """
        Write ID list
        :param content: list to write
        :param file_out: Output file
        :param split_id: Indicate whether Id should be split (and only 1st part is considered)
        :return:
        """
        with open(file_out, 'w') as out:
            for c in content:
                if split_id is not None:
                    c_new = c.split(split_id)[0]
                else:
                    c_new = c
                out.write('{}\n'.format(c_new))

    @staticmethod
    def write_dict_dict(file_out, content, keys, header=None, file_action='w'):
        """
        Write dictionary of dictionary (with header) to file
        :param file_out: Output file
        :param content: Dictionary to write
        :param keys: Keys of the inner dictionary
        :param header: Optional (header to write as 1st line)
        :param file_action: Whether content should be written into new file (overwrite existing)
                    or appended to existing file
        :return:
        """

        with open(file_out, file_action) as out:
            if header is not None:
                out.write(header)
            for k in content.keys():
                c = k.split('-')[0]
                out.write('{}\t{}'.format(c, k))
                for key in keys:
                    out.write('\t{}'.format(content[k][key]))
                out.write('\n')

    @staticmethod
    def write_dict_dict_dict(file_out, content, keys, header=None, file_action='w'):

        with open(file_out, file_action) as out:
            if header is not None:
                out.write(header)

            for k in content.keys():
                for s in content[k].keys():
                    c = s.split('-')[0]
                    out.write('{}\t{}\t{}'.format(c, k, s))
                    for key in keys:
                        out.write('\t{}'.format(content[k][s][key]))
                    out.write('\n')

    @staticmethod
    def write_dict_list(file_out, content, header=None, file_action='w'):

        with open(file_out, file_action) as out:
            if header is not None:
                out.write(header)

            for k in content.keys():
                for el in content[k]:
                    out.write('{}\t{}\n'.format(k, el))


class Utilities(object):

    @staticmethod
    def get_relevant_ids(uni_ids):
        relevant_ids = []
        for u in uni_ids:
            if '_' not in u:
                relevant_ids.append(u)
        return relevant_ids

    @staticmethod
    def combine_dicts(dict1, dict2):
        for k in dict2.keys():
            if k in dict1.keys():
                dict1[k] += dict2[k]
            else:
                dict1[k] = dict2[k]

        return dict1

    @staticmethod
    def create_random_clustering(sequences, cluster_sizes, num_outliers, outlier_probas):
        clusters = defaultdict(list)
        outliers = []
        probabilities = dict()

        # randomize sequences and outlier probabilities
        rnd_sequences = random.sample(sequences, len(sequences))
        rnd_probas = random.sample(outlier_probas, len(outlier_probas))

        # classify first num_outliers sequences as outliers
        if num_outliers > 0:
            outliers = rnd_sequences[0: num_outliers]

        # cluster remaining sequences in clusters of cluster_sizes
        cluster_sequences = rnd_sequences[num_outliers:]
        last_ind = 0
        for i, c in enumerate(cluster_sizes):
            seqs = cluster_sequences[last_ind: last_ind + c]
            last_ind += c
            clusters[i] = seqs

        # assign each sequence a random probability
        for i, p in enumerate(rnd_probas):
            probabilities[sequences[i]] = p

        return clusters, outliers, probabilities


class Npy2Npz(object):

    @staticmethod
    def get_dataset(npz_path, use_cath=True):
        raw_data_path = npz_path.parent / npz_path.name.replace('.npz', '.npy')
        ids_path = npz_path.parent / npz_path.name.replace('.npz', '_ids.txt')

        # Load raw data from two seperate files if already created
        # IDs are stored as txt and embeddings are stored as npz (uncompressed)
        # Row-Indices in .npz correspond to line number in IDs file to keep track of ID-Embedding pairs
        if raw_data_path.is_file() and ids_path.is_file():
            raw_data = np.load(raw_data_path)
            with open(ids_path, 'r') as id_f:
                ids = [line.strip() for line in id_f]
            dataset = {seq_id: np.expand_dims(raw_data[idx], axis=0)
                       for idx, seq_id in enumerate(ids)}

        # Otherwise, if only npy file (compressed dictionary) exists:
        # Load dictionary, split Key/Value pairs and write Keys as txt and
        # concatenated embeddings as npz
        else:
            dataset = dict(np.load(npz_path, mmap_mode='r'))
            ids, raw_data = zip(*dataset.items())

            Npy2Npz._write_files(ids_path, raw_data_path, ids, raw_data, use_cath)
            dataset = {seq_id: np.expand_dims(embd, axis=0)
                       for seq_id, embd in dataset.items()}

        return dataset

    @staticmethod
    def get_dataset_uncompressed(npy_file, id_file):
        raw_data = np.load(npy_file)
        with open(id_file, 'r') as read_in:
            ids = [line.strip() for line in read_in]
            # ids = [next(read_in).strip() for line in range(100000)]

        dataset = {seq_id: np.expand_dims(raw_data[idx], axis=0)
                   for idx, seq_id in enumerate(ids)}

        return dataset

    @staticmethod
    def get_dataset_uncompressed_partial(npy_file, id_file, start_index, end_index):
        raw_data = np.load(npy_file)
        with open(id_file, 'r') as read_in:
            # ids = [line.strip() for line in read_in]
            ids = [next(read_in).strip() for line in range(start_index, end_index)]

        dataset = {seq_id: np.expand_dims(raw_data[idx], axis=0)
                   for idx, seq_id in enumerate(ids)}

        return dataset

    @staticmethod
    def write_dataset_speedup(path, data, use_cath=False):
        ids, raw_data = zip(*data.items())

        raw_data_path = path.parent / path.name.replace('.npz', '.npy')
        ids_path = path.parent / path.name.replace('.npz', '_ids.txt')

        Npy2Npz._write_files(ids_path, raw_data_path, ids, raw_data, use_cath)

    @staticmethod
    def _write_files(ids_path, raw_data_path, ids, raw_data, use_cath=False):

        with open(ids_path, 'w+') as id_f:
            for seq_id in ids:
                if use_cath:
                    seq_id = seq_id.split('|')[2].split('/')[0]
                id_f.write(seq_id + '\n')
        np.save(raw_data_path, raw_data)
