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
            else:
                os.remove(os.path.join(cfg.pathtodata, cfg.data, dirname, filename))


def create_dir():
    if not os.path.exists(os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers')):
        mkdirCmd1 = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers'))
        os.system(mkdirCmd1)
        mkdirCmd2 = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers', 'output'))
        os.system(mkdirCmd2)
        mkdirCmd3 = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers', 'temp'))
        os.system(mkdirCmd3)
        mkdirCmd3 = "mkdir %s" % (os.path.join(cfg.pathtoxp, cfg.xp_name, 'temp'))
        os.system(mkdirCmd3)


changefna2fa()
create_dir()
