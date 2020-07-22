#TODO create dir for experiment
#TODO create function to rename files from .fna to .fa
import os
import config as cfg

def changefna2fa():
    for dirname in os.listdir(cfg.pathtoxp + 'data/'+cfg.data):
        for filename in os.listdir(cfg.pathtoxp + 'data/'+cfg.data + dirname):
            print(filename[-4:])
            if filename[-4:]=='.fna':
                os.rename(cfg.pathtoxp+'data/'+cfg.data+dirname+'/'+filename, cfg.pathtoxp+'data/'+cfg.data+dirname+'/'+filename[:-4]+'.fa')
            else:
                os.remove(cfg.pathtoxp+'data/'+cfg.data+dirname+'/'+filename)


changefna2fa()