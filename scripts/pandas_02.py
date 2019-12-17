import numpy as np
import pandas as pd

x = np.arange(0, 10)

# Create date frame
data = {'x': x, 'sin': np.sin(x), 'cos': np.cos(x)}
trigonometry = pd.DataFrame(data, columns=['x', 'sin', 'cos'])
print(trigonometry)

# Read data from csv
csvdata = pd.read_csv('pandas_02.csv')
print(csvdata.head())

# Write data to csv
trigonometry.to_csv('trigonometry.csv')

# Write data to excel
trigonometry.to_excel('trigonometry.xlsx', index=False)
trigonometry.info()
print(trigonometry.dtypes)
