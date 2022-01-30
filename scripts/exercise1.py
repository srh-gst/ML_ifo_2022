# Exercise 1

# 0. packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 1. Tips
tips = sns.load_dataset("tips")

tips = tips.replace("Mon", "Monday")
tips = tips.replace("Tues", "Tuesday")
tips = tips.replace("Wed", "Wednesday")
tips = tips.replace("Thur", "Thursday")
tips = tips.replace("Fri", "Friday")
tips = tips.replace("Sat", "Saturday")
tips = tips.replace("Sun", "Sunday")

tips

fig = sns.relplot(data=tips, x = "total_bill", y = "tip", hue="day", col="sex")
fig.savefig("./output/tips.pdf")

# 2. Occupations

FNAME = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"
df = pd.read_csv(FNAME, sep="|")
df.head(25)
df.tail(10)
df.info()
occ_count = df.value_counts("occupation")
occ_count = pd.DataFrame(occ_count)
occ_count = occ_count.reset_index()
occ_count.columns = ['occupation', 'counts']
np.count_nonzero(occ_count["occupation"]) #21
occ_count.loc[occ_count['counts'] == occ_count['counts'].max()]["occupation"] #student
occ_count.set_index(["occupation"],inplace=True)
occ_count = occ_count.sort_index()
occ_count
fig, ax = plt.subplots()
occ_count.plot.bar(ax=ax)
fig.tight_layout(pad=2)
ax.set_xlabel("Frequency of occupation")
fig.savefig("./output/occupations.pdf")

# 3. Iris

FNAME =  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colnames=['sepal_length', 'sepal_width', 'petal_length', 'petal_witdth', 'class']
df = pd.read_csv(FNAME, sep=",", names = colnames, header = None)
df.head()
df.info()
df.iloc[0:29,2]=np.NaN
df.iloc[0:29, 2]
df = df.replace(np.NaN, 1)
df.to_csv('./output/iris.csv', sep=',')






