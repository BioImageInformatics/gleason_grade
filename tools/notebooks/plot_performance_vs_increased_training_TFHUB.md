---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd
import os

%matplotlib inline
from matplotlib import pyplot as plt
```

```python
datadir = '../tfhub_mobilenet'
pcts = [ '01pct', '02pct', '05pct', '10pct', '25pct', '75pct' ]
positions = [0.1, 0.2, 0.5, 0.1, 0.25, 0.75]
datafiles = [os.path.join(datadir, pct, 'processed_perf.tsv') for pct in pcts]
```

```python
df = pd.read_csv(datafiles[0], header=0, index_col=0, sep='\t')
df.head()
```

```python
col = 'EpitheliumF1'
def get_performance(fpath, col=col):
    print(fpath)
    df = pd.read_csv(fpath, header=0, index_col=0, sep='\t')
    f1 = df[col].values
    return f1

f1_scores = [get_performance(fpath) for fpath in datafiles]
```

```python
plt.figure(figsize=(4,3), dpi=180)
_ = plt.boxplot(f1_scores)
plt.ylabel('F1 Score', fontsize = 18)
plt.xlabel('% training images', fontsize = 18)
plt.title('MobileNet V2', fontsize=20)
_ = plt.xticks(np.arange(1, len(f1_scores)+1), ['{:2.0f}%'.format(x*100) for x in positions])
```

```python

```
