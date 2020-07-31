# TODO create dir for experiment
# TODO create function to rename files from .fna to .fa
import os
import config as cfg


def changefna2fa():
    for dirname in os.listdir(os.path.join(cfg.pathtodata, cfg.data)):
        for filename in os.listdir(os.path.join(cfg.pathtodata, cfg.data + dirname)):
            print(filename[-4:])
            if filename[-4:] == '.fna':
                os.rename(os.path.join(cfg.pathtodata, cfg.data + dirname, filename),
                          os.path.join(cfg.pathtodata, cfg.data, dirname, filename[:-4] + '.fa'))
            elif filename[-4:]=='.tab':
                os.remove(os.path.join(cfg.pathtodata, cfg.data, dirname, filename))


def create_dir():
    if not os.path.exists(os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers')):
        mkdirCmd1 = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers'))
        os.system(mkdirCmd1)
        # mkdirCmd2 = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers', 'output'))
        # os.system(mkdirCmd2)
        # mkdirCmd3 = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers', 'temp'))
        # os.system(mkdirCmd3)
        mkdirCmd3 = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, 'temp'))
        os.system(mkdirCmd3)

def write_kover_metadata_files():
    genomedata=open(os.path.join(cfg.pathtoxp, 'GenomeData.txt'),'w')
    metadata=open(os.path.join(cfg.pathtoxp, 'metadata.txt'),'w')
    for dirname in os.listdir(os.path.join(cfg.pathtodata, cfg.data)):
        for filename in os.listdir(os.path.join(cfg.pathtodata, cfg.data + dirname)):
            if dirname=='Resistant':
                genomedata.write(filename[:-3]+'\t'+str(os.path.join(cfg.pathtodata, cfg.data+dirname, filename))+'\n')
                metadata.write(filename[:-3]+'\t'+'1\n')
            elif dirname=='Susceptible':
                genomedata.write(filename[:-3]+'\t'+str(os.path.join(cfg.pathtodata, cfg.data+dirname, filename))+'\n')
                metadata.write(filename[:-3]+'\t'+'0\n')

#changefna2fa()
#create_dir()
write_kover_metadata_files()