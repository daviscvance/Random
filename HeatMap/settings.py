import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import datetime
import gmaps
import sys
import os
import re

def warn(*args, **kwargs):
        pass
warnings.warn = warn

# Style settings
#plt.style.available
plt.style.use('fivethirtyeight')
sns.set_context('talk')

# Option settings
#pd.describe_option('display')
pd.options.display.max_rows = 20
pd.options.display.max_columns = 99
pd.options.display.max_colwidth = 99
