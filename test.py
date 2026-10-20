#import random
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

data = pd.read_csv("2019.csv")
