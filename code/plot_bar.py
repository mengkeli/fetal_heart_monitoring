import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('../data/zero_count.csv')
#table=pd.crosstab(pd.DataFrame(np.arange(24360)),data)
#table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

data.plot(kind='bar', stacked=True)
plt.title('zero_value in raw data')
plt.xlabel('samples')
plt.ylabel('zero_value numbers in each sample')
plt.savefig('zero_count_in_raw_data')

plt.rc("font", size=14)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
sns.axes_style()
sns.despine(left=True,)
sns.set(rc={"figure.figsize": (60, 6)},style='white');
ax = sns.tsplot(data=sample1)

x = np.linspace(0, 2402, 1)
y = np.array(sample1)
tck = interpolate.splrep(x, y)
y_bspline = interpolate.splev(x_new, tck)