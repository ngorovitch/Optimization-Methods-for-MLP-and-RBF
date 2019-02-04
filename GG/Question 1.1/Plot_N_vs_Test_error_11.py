# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Make a data frame
df=pd.DataFrame({'N': np.asarray([1, 2, 3, 4, 8, 16, 20, 30, 50]), 
                 'Final Training Error': np.asarray([0.347974412, 0.221202762, 0.126834688, 0.04269635, 0.017717824, 0.001390667, 0.000960427, 0.000993104, 0.000835254900372628]),
                 'Final Test Error': np.asarray([0.427243745, 0.292750155, 0.161141306, 0.044514888, 0.038716909, 0.004798232, 0.004973978, 0.00572255, 0.00433422267017162])
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
