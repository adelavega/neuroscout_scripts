from coda import BIDSEventReader
from coda import EventTransformer
from random import randint
import pandas as pd

reader = BIDSEventReader(base_dir='/Users/alejandro/datasets/ds000102/')
events = reader.read(subject='01', run='run-1').drop(['type', 'modality', 'task', 'run'], axis=1)

all_events = []
for sub in range(0, 4):
	e = events.copy()
	e['amplitude'] = [randint(0, 10) for i in range(0, e.shape[0])]
	e['subject'] = sub
	all_events.append(e)

all_pd = pd.concat(all_events)

for sub, data, in all_pd.groupby('subject'):
	tdata = EventTransformer(data).data                                                                                                                                   
	tdata['subject'] = sub
	all_pd_dense.append(tdata)

all_pd_dense = pd.concat(all_pd_dense)

all_pd_dense = pd.melt(all_pd_dense, id_vars=['onset', 'subject'], var_name='condition', value_name='amplitude')
