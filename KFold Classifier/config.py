import configparser
import argparse
import numpy as np

config = configparser.ConfigParser()
ues = {}
for i in range (1,41):
    a = 'ue'+str(i)+'.txt'
    with open ("//lab//users//Cruz//vivian//LearnRAN//Data//"+a) as b:
        ues[i]= eval('['+b.read()+']')
database = ues
config['data_ues'] = database
