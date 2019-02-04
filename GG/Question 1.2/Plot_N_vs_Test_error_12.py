# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Make a data frame
df=pd.DataFrame({'N': np.asarray([1, 2, 3, 4, 8, 16, 20, 30, 50]), 
                 'Final Training Error': np.asarray([0.326113452, 0.302117917, 0.285639874, 0.242497602, 0.15386382, 0.018255197, 0.005182247, 0.05184713, 0.00187615852407706]),
                 'Final Test Error': np.asarray([0.369147178, 0.344148177, 0.356000497, 0.286024058, 0.180657433, 0.025534061, 0.010293475, 0.12403595, 0.00639321194676693])
                })
 
# style
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set2')
 
# multiple line plot
num=0
for column in df.drop('N', axis=1):
    num+=1
    plt.plot(df['N'], df[column], marker='', color=palette(num), linewidth=2, alpha=1, label=column)
 
# Add legend
plt.legend(loc=1, ncol=1)
 
# Add titles
plt.title("N vs Training/Test Error", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("N")
plt.ylabel("Error")
plt.show()
