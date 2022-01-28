import pandas as pd

data1 = [("wednesday", "good"), ("thursday", "good"), ("friday", "bad")]
df = pd.DataFrame(data1)

df.head()
df.to_latex("latex_test.tex", index = False)

x = "9"
y = 1
z = x + y



