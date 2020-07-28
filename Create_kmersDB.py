import os
import pandas as pd
import pickle
import config as cfg
import fastparquet


class KmerExtractionAndCount(object):
    """
    Class used in KmersCount2Dataframe
    Extract kmers counts using gerbil https://github.com/uni-halle/gerbil

    Marius Erbert, Steffen Rechner, and Matthias Müller-Hannemann, Gerbil: A fast and memory-efficient k-mer counter
    with GPU-support, Algorithms for Molecular Biology (2017) 12:9, open access.
    """

    def __init__(self, fastaname):

        self.pathtofasta = os.path.join(cfg.pathtodata,cfg.data, fastaname)
        self.strainnumber = self.pathtofasta.split('/')[-1][:-3]
        self.label = self.pathtofasta.split('/')[-2]

        self.available_commands = ['parse_kmers']
        self.len_kmers = cfg.len_kmers
        self.min_abundance = cfg.min_abundance

        self.pathtotemp = os.path.join(cfg.pathtoxp, cfg.xp_name, 'temp/')
        self.pathtosavetemp = os.path.join(cfg.pathtoxp, cfg.xp_name , 'kmers/temp', self.strainnumber)
        self.pathtosave = os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers/output', self.label+ self.strainnumber)


    def parse_kmers(self):
        kmerCmd = "gerbil -k %d  -l %d %s %s %s" % (
            self.len_kmers, self.min_abundance, self.pathtofasta, self.pathtotemp, self.pathtosavetemp)
        os.system(kmerCmd)  # default minimal abundance of kmers=3, can be changed with "-l newvalue"
        tofastaCmd = "toFasta %s %d %s" % (self.pathtosavetemp, self.len_kmers, self.pathtosave)
        os.system(tofastaCmd)

        self.kmer_counts = {}
        with open(self.pathtosave, 'r') as fasta:
            lines = fasta.readlines()
            self.kmer_counts['strain'] = self.strainnumber
            for kmercount, kmerID in zip(*[iter(lines)] * 2):
                try:
                    count = int(kmercount[1:])
                    ID = kmerID[:-1]
                    self.kmer_counts[ID] = count
                except:
                    print(kmercount, kmerID)

        self.kmer_counts['label'] = self.label


class KmersCounts2Dataframe(object):
    """
    This class allows us to iterate over fastas file, count kmers using the class KmerExtraction and Count and create a
    pandas dataframe using these kmers counts
    """

    def __init__(self):
        self.kmerdicts = 0


    def iteratefastas(self):
        self.kmerdicts = []
        for dirname in os.listdir(os.path.join(cfg.pathtodata, cfg.data)):
            for filename in os.listdir(os.path.join(cfg.pathtodata, cfg.data, dirname)):
                self.kmer = KmerExtractionAndCount(dirname + '/' + filename)
                self.kmer.parse_kmers()
                self.kmerdicts.append(self.kmer.kmer_counts)

        with open(os.path.join(cfg.pathtoxp  ,cfg.xp_name, 'kmerdicts.pkl'), 'wb') as f:
            pickle.dump(self.kmerdicts, f)

    def create_dataframe(self):
        if not self.kmerdicts:
            with open(os.path.join(cfg.pathtoxp , cfg.xp_name , 'kmerdicts.pkl'), 'rb') as f:
                self.kmerdicts = pickle.load(f)

        self.kmerdicts = pd.DataFrame(self.kmerdicts)
        self.kmerdicts = self.kmerdicts.fillna(0)

        # with open(os.path.join(cfg.pathtoxp , cfg.xp_name , 'kmers_DF.pkl'), 'wb') as f:
        #     pickle.dump(self.kmerdicts, f)
        fastparquet.write(os.path.join(cfg.pathtoxp , cfg.xp_name , 'kmers_DF.parq'), self.kmerdicts)

    def clean_temp_directories(self):
        cleankmertempcmd="rm -rf %s" % (self.kmer.pathtotemp)
        os.system(cleankmertempcmd)
        cleantempcmd="rm -rf %s" % (self.kmer.pathtosavetemp)
        os.system(cleantempcmd)



# kmergenerator = KmersCounts2Dataframe()
# kmergenerator.iteratefastas()
# kmergenerator.create_dataframe()
# kmergenerator.clean_temp_directories()
