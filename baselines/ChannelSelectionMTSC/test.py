import sys 
import os
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_io import load_from_tsfile_to_dataframe

from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.classification.dictionary_based import MUSE
import time
import numpy as np
from TSpy.label import compact

from src.kmeans import kmeans
from  src.classelbow import ElbowPair # ECP
from src.elbow import elbow # ECS..

import pandas as pd
import os

# selects class prototype
center='mad' # options: mean, median
#elb = kmeans()
elb  = elbow(distance = 'eu', center=center) # Select elbow class sum
#elb = ElbowPair(distance = 'eu', center=center) # Selects elbow class Pair
#elb = None

    
model = Pipeline(
        [
        ('classelbow', elb),
        ('rocket', Rocket(random_state=0,normalise=False)),
        ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),normalize=True ))
        #('SEQL', MrSEQLClassifier()),
        #('weasel_muse', MUSE(random_state=0)),
        ],
        #verbose=True,
        )

script_path = os.path.dirname(__file__)
train = "data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.ts"
test = "data/ArticularyWordRecognition/ArticularyWordRecognition_TEST.ts"
train = os.path.join(script_path, train)
test = os.path.join(script_path, test)

train_x, train_y = load_from_tsfile_to_dataframe(train)
test_x, test_y = load_from_tsfile_to_dataframe(test)

def find_cut_points_from_state_seq(state_seq):
    # the last element in cut_point_list is the length of state_seq.
    cut_point_list = []
    c = state_seq[0]
    for i, e in enumerate(state_seq):
        if e == c:
            pass
        else:
            cut_point_list.append(i)
            c = e
    cut_point_list.insert(0, 0)
    cut_point_list.append(i+1)
    return cut_point_list

# train_x = train_x.to_numpy()
print(type(train_x))
print(type(train_x.iloc[0][0]))
print(train_x.shape)
# print(type(train_x[0][0]))

label = pd.read_csv(os.path.join(script_path, '../../data/MoCap/amc_86_08.csv')).to_numpy()[:,4]
data = pd.read_csv(os.path.join(script_path, '../../data/MoCap_csv/86_08.csv')).to_numpy()[:-1]

cut_points = find_cut_points_from_state_seq(label)
print(cut_points)
segments = [data[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points)-1)]
print(len(segments))
# padding to the same length
# max_len = max([len(segment) for segment in segments])
# segments = [np.pad(segment, ((0,max_len-len(segment)),(0,0)), 'constant') for segment in segments]
# cut to the same length
min_len = min([len(segment) for segment in segments])
segments = [segment[:min_len] for segment in segments]
new_segments = []
for seg in segments:
    # convert each seg to a list of tuple
    new_seg = [pd.Series(e) for e in seg.T]
    new_segments.append(pd.Series(new_seg))
    # new_seg = [tuple(e) for e in seg.T]
    # new_segments.append(pd.Series(new_seg))

segments = pd.DataFrame(new_segments)
print(type(segments))
print(type(segments.iloc[0][0]))
print(segments.shape)

print(compact(label))
elb.fit(segments, compact(label))
print(elb.relevant_dims)

import matplotlib.pyplot as plt
plt.plot(data[:,5])
plt.plot(data[:,3])
plt.plot(data[:,4])
plt.savefig('test.png')