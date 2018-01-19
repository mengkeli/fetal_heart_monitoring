import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('../data/zero_count.csv')
table=pd.crosstab(pd.DataFrame(np.arange(24360)),data.zero_count)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('ero count in original data')
plt.xlabel('samples')
plt.ylabel('zero_value number in each sample')
plt.savefig('zero_count_in_raw_data')