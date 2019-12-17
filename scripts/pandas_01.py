import pandas as pd

# Add an array with index names
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'],
              index=['A', 'Z', 'C', 'Y', 'E'])

# Use a dictionary to create a series
stocks = pd.Series({'MO': 55, 'KO': 40, 'SBUX': 39, 'BMW': 67, 'T': 12})

# Select multiple values per index
print(stocks[['MO', 'SBUX', 'KO']])

# Select values conditionally
print(stocks[stocks < 30])

# Modify conditionally selected values
print('Old values\n', stocks)
stocks[stocks <= 40] *= 1.1
print('New values\n', stocks)

# Check content
print('MO' in stocks)
print('Döner' in stocks)

# Operators
print(stocks * 0.85)

stocksDiff = pd.Series({'KO': 0.4, 'SBUX': 3, 'ABBV': 2, 'T': -2})
stocksResult = stocks + stocksDiff
print(stocksResult)
print(stocksResult.notnull())
