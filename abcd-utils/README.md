# abcd_utils
Collection of utility scripts to explore and process ABCD data

## Installation
To install with `pip`, run:
```
pip install git+ssh://git@github.com:rauschecker-sugrue-labs/abcd-utils.git
```

## Usage
```py
import pandas as pd
from abcd import utils as ut
from abcd.data_loader import DataLoader

DL = DataLoader(version='4.0')
# Find variable (lookup by element name, in description, and alias)
DL.find_variable('toolbox')
DL.find_variable('gender')
# Get one or more elements data
DL.get_variable_data('interview_age', try_substitution=True)
df1 = DL.get_variables_data(['sex','interview_age', 'fsqc_qc'])

df2 = pd.read_csv('my_abcd_subset.csv')
# Smartly merge ABCD related datasets
df = ut.abcd_merge([df1, df2])
df.head()
```
