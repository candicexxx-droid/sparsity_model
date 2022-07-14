import numpy as np
from utils import *

#random initializations


cluster_a = np.array([0.1, 0.4, 0.3, 0.5, 0.25])
cluster_b = np.array([0.37, 0.42, 0.15, 0.2, 0.7])

# experiment_1 = np.array([True, False, False, True, True])

data=DatasetFromFile('sanity_check')
experiments = data.x.cpu().detach().numpy()
print('import done')