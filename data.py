import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config as cfg


class Kmer_parser(object):
    """
    Object called in KmersCount2Dataframe class
    Extract kmers using terminal softwares (DSK or Gerbil)
    """

    def __init__(self, fastaname):
        self.pathtofasta = os.path.join(cfg.pathtodata, cfg.data, fastaname)
        self.strainnumber = self.pathtofasta.split('/')[-1][:-3]
        self.label = fastaname[:5]
        self.available_commands = ["parse_kmers_dsk", "parse_kmers_gerbil"]
        self.len_kmers = cfg.len_kmers
        self.min_abundance = cfg.min_abundance
        self.pathtotemp = os.path.join(cfg.pathtoxp, cfg.xp_name, "temp", self.label + self.strainnumber)
        self.pathtosave = os.path.join(cfg.pathtoxp, cfg.xp_name, "kmers", self.label + self.strainnumber)

    def parse_kmers_gerbil(self):
        kmerCmd = "gerbil -k %d  -l %d %s %s %s" % (self.len_kmers, self.min_abundance, self.pathtofasta, self.pathtotemp, self.pathtosavetemp)
        os.system(kmerCmd)
        tofastaCmd = "toFasta %s %d %s" % (self.pathtosavetemp, self.len_kmers, self.pathtosave)
        os.system(tofastaCmd)

    def count_kmers_gerbil(self):
        self.kmer_counts = {}
        with open(self.pathtosave, "r") as fasta:
            lines = fasta.readlines()
            self.kmer_counts["strain"] = self.strainnumber
            for kmercount, kmerID in zip(*[iter(lines)] * 2):
                try:
                    count = int(kmercount[1:])
                    ID = kmerID[:-1]
                    self.kmer_counts[ID] = count
                except:
                    print(kmercount, kmerID)

    def parse_kmers_dsk(self):
        kmerCmd = "dsk -file %s -out %s -kmer-size %d -abundance-min 1 -verbose 0" % (self.pathtofasta, self.pathtotemp, self.len_kmers)
        os.system(kmerCmd)
        outputCmd = "dsk2ascii -file %s -out  %s" % (self.pathtotemp, self.pathtosave)
        os.system(outputCmd)

    def count_kmers_dsk(self):
        self.kmer_counts = {}
        with open(self.pathtosave, "r") as fasta:
            lines = fasta.readlines()
            for line in lines:
                try:
                    [ID, count] = line.split(" ")
                    self.kmer_counts[str(ID)] = int(count)
                except:
                    print("line = " + line)


class Kmercount_to_matrix(object):
    """
    This class allows us to iterate over fastas file, count kmers using the class KmerExtractionandCount and create a
    scipy sparse csr matrix or a pandas dataframe using these kmers counts
    """

    def __init__(self):
        self.kmerdicts = 0

    def create_sparse_coos(self):
        print("*** Creating matrix ***")
        if not self.kmerdicts:
            print("loading kmers dictionary")
            with open(os.path.join(cfg.pathtoxp, cfg.xp_name, "kmerdicts.pkl"), "rb") as f:
                self.kmerdicts = pickle.load(f)

        self.strain_to_index = {strain: i for i, strain in zip(range(len(self.strains)), self.strains)}
        self.kmer_to_index = {kmer: i for i, kmer in enumerate(self.kmerdicts)}

        rows = []
        columns = []
        data = []

        for kmer in self.kmerdicts:
            for strain in self.kmerdicts[kmer]:
                rows.append(self.strain_to_index[strain])
                columns.append(self.kmer_to_index[kmer])
                data.append(self.kmerdicts[kmer][strain])
        del self.kmerdicts
        return rows, columns, data

    def populate_sparse_matrix(self, rows, cols, data):
        n_strains = len(self.strains)
        self.mat = sp.csr_matrix((data, (rows, cols)), shape=(n_strains, len(self.kmer_to_index)), dtype=np.int8)

        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, "kmers_mats.pkl"), "wb") as f:
            pickle.dump([self.mat, self.labels, self.strain_to_index, self.kmer_to_index], f, protocol=4)

    def extend_kmerdicts(self, kmer):
        for key in kmer.kmer_counts.keys():
            if key in self.kmerdicts:
                self.kmerdicts[key][kmer.strainnumber] = int(kmer.kmer_counts[key])
            else:
                self.kmerdicts[key] = {kmer.strainnumber: int(kmer.kmer_counts[key])}

    def create_dataframe(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        print("*** Creating dataframe ***")
        if not self.kmerdicts:
            with open(os.path.join(cfg.pathtoxp, cfg.xp_name, "kmerdicts.pkl"), "rb") as f:
                self.kmerdicts = pickle.load(f)

        self.kmerdb = pd.DataFrame(self.kmerdicts)

        table = pa.Table.from_pandas(self.kmerdicts.transpose)
        pq.write_table(table, os.path.join(cfg.pathtoxp, cfg.xp_name, "kmers_DF.parquet"))

    def clean_temp_directories(self, kmer):
        cleankmertempcmd = "rm -rf %s" % (kmer.pathtotemp)
        os.system(cleankmertempcmd)
        # cleantempcmd="rm -rf %s" % (self.kmer.pathtosavetemp)
        # os.system(cleantempcmd)

    def run(self):
        self.kmerdicts = {}
        self.labels = []
        self.strains = []
        #for dirname in os.listdir(os.path.join(cfg.pathtodata, cfg.data)):
        for filename in os.listdir(os.path.join(cfg.pathtodata, cfg.data)):
            kmer = Kmer_parser(os.path.join(filename))
            kmer.parse_kmers_dsk()
            kmer.count_kmers_dsk()
            self.strains.append(kmer.strainnumber)
            self.labels.append(kmer.label)
            self.extend_kmerdicts(kmer)
            self.clean_temp_directories(kmer)

        rows, cols, data = self.create_sparse_coos()
        self.populate_sparse_matrix(rows, cols, data)




# kmergenerator = KmersCounts2Dataframe()
# kmergenerator.iteratefastas()
# kmergenerator.create_sparse_matrix()
