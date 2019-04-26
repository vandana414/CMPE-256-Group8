import pandas as pd
import numpy as np
import pickle

class labels:
    def createLabel(labels):
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR == 'Arson']=0
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR == 'Lightning']=1
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR == 'Debris Burning']=2
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR == 'Campfire']=3
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR =='Children']=3
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR =='Fireworks']=3
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR =='Powerline']=3
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR =='Railroad']=3
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR =='Smoking']=3
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR =='Structure']=3
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR =='Equipment Use']=3
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR == 'Miscellaneous']=4
        labels['STAT_CAUSE_DESCR'][labels.STAT_CAUSE_DESCR == 'Missing/Undefined']=4
        return labels