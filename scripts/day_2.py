import panda as pd

data1 = [("wednesday", "good"), ("thursday", "good"), ("friday", "bad")]
df= pd.DataFrame(data1)

df.head()
df.to_latex("latex_test.tex", index = False)



