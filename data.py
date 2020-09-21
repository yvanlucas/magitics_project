import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
#from Bio import SeqIO

import config as cfg


class Kmer_parser(object):
    '''
    Extract kmers using terminal softwares (DSK or Gerbil)
    class called in KmersCount2Dataframe class
    '''
    def __init__(self, fastaname):
        '''

        :param fastaname: name of the file from which kmers are parsed
        '''
        self.pathtofasta = os.path.join(cfg.pathtodata, cfg.data, fastaname)
        self.strainnumber = self.pathtofasta.split('/')[-1][:-3]
        self.label = fastaname[:5]
        self.available_commands = ["parse_kmers_dsk", "parse_kmers_gerbil"]
        self.len_kmers = cfg.len_kmers
        self.min_abundance = cfg.min_abundance
        self.pathtotemp = os.path.join(cfg.pathtoxp, cfg.xp_name, "temp", self.label + self.strainnumber)
        self.pathtosave = os.path.join(cfg.pathtoxp, cfg.xp_name, "kmers", self.label + self.strainnumber)

    def parse_kmers_gerbil(self):
        '''

        :return:
        '''
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
        print(self.pathtofasta)
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

        mkdircmd = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id))
        os.system(mkdircmd)

        with open(os.path.join(cfg.pathtoxp, cfg.xp_name,cfg.id, "kmers_mats.pkl"), "wb") as f:
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
            #self.clean_temp_directories(kmer)

        rows, cols, data = self.create_sparse_coos()
        self.populate_sparse_matrix(rows, cols, data)


class parse_genes_limits(object):
    def __init__(self):
        return

    def get_genes_limit(self, pathtofile):
        with open(pathtofile, 'r') as f:
            lines=f.readlines()[1:]
        count_contigs=0
        dic_limits={}
        for line in lines[:3]:
            line=line.split('\t')
            print(line)
            if line[9] == '1':
                count_contigs +=1
            pgfam=line[-4]
            limits = [(int(line[9]), int(line[10])), count_contigs]

            if pgfam in dic_limits:
                dic_limits[pgfam].append(limits)
            else:
                dic_limits[pgfam] = [limits]
        return dic_limits

    def extract_sequence(self, pathtofile):
        with open(pathtofile, 'r') as f:
            file=f.readlines()
        contigs=[[]]
        skipfirst=1
        for i in file:
            if skipfirst==0:
                contigs[-1].append(str(i)[:-1])
            skipfirst=0
            if len(i)==1:
                contigs += [[]]
                skipfirst=1
        return contigs

    def extract_genes_from_seq(self, contigs, dic_limits):
        dic_genes={}
        for pgfam in dic_limits:
            dic_genes[pgfam]=[]
            for limits in dic_limits[pgfam]:
                geneseq=contigs[limits[1]][limits[0][0]:limits[0][1]]
                dic_genes[pgfam].append(geneseq)
        return dic_genes

    def kmers_within_gene(self, geneseq, len_kmers=31):
        ls_kmers=[]
        for i in range(len(geneseq)-len_kmers):
            ls_kmers.append(geneseq[i:i+len_kmers])
        return ls_kmers

    def build_kmer_gene_dict(self, dic_kmer_gene, dic_gene):
        """
        dic_kmer_gene: {kmer:[pgfam1, pgfam2, ..]}
        dic_gene: {pgfam:[seq1, seq2, ..]}
        """
        for pgfam in dic_gene.keys():
            for seq in dic_gene[pgfam]:
                ls_kmers=self.kmers_within_gene(seq)
                for kmer in ls_kmers:
                    if kmer in dic_kmer_gene:
                        dic_kmer_gene[kmer].append(pgfam)
                    else:
                        dic_kmer_gene[kmer] = [pgfam]
        return dic_kmer_gene

    def get_genes_from_kmers(self):
        """
        Method used to understand results
        """
        return

    def run(self):
        dic_kmer_gene={}
        for filename in os.listdir(os.path.join(cfg.pathtodata, cfg.data))[::2]:
            limitfile=filename
            seqfile=filename
            dic_limits=self.get_genes_limit(limitfile)
            contigs=self.extract_sequence(seqfile)
            dic_genes=self.extract_genes_from_seq(contigs, dic_limits)
            dic_kmer_gene=self.build_kmer_gene_dict(dic_kmer_gene, dic_genes)

        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id,'dic_kmer_gene.pkl'),'w') as f:
            pickle.dump(dic_kmer_gene, f)



gene_limits=parse_genes_limits(os.path.join(cfg.pathtodata, cfg.data, '287.846'))

gene_limits.run()
