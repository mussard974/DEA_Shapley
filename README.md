# DEA_Shapley
---
Compute DEA with the directional distance function (DDF) and return Shapley scores for each DMU
---

### DEA_Shapley is a method to fit DDF DEA models with Shapley contributions

In this package, we find:

  * Computation of DDF by DEA with CRS, DRS, VRS, IRS
  * Choose your own direction g for the DDF
  * Computation for one or multi-outputs
  * Shapley contributions
  * Equal Surplus contributions (ES)
  * Feature importance contributions

### Libraries
* numpy
* pandas
* pulp

```python
pip install pulp
```

### Import Shapley

```python
from Shapley import ShapleyModel
```

### Import data 

```python
# Data
import pandas as pd
import numpy as np
base = pd.read_excel('data.xlsx')
base.head()
```

### Organize the data (X = inputs y = outputs)

```python
y1 = 1000 - cepej['disposition time'].to_numpy()
y2 = cepej['clearance rate'].to_numpy()
Y = np.column_stack((y1, y2))
X = cepej[[
       "number_judges",
       "number_no-judges",
       "Information tools",
       "Tools of communication"
       ]].to_numpy()
```

### Parameters of the model

```python
outputs = "multi_dimensions"  # "one_dimension" for just one output
g_inputs = np.ones((X.shape[1]))
g_outputs = np.ones((Y.shape[1]))
constraint = "DRS"
```

### Instantiate and fit the model (the function returns an array with all $s$-curves)

```python
from Shapley import ShapleyModel
model = ShapleyModel(outputs = "multi-outputs", constraint = "DRS", g_inputs = np.ones((X.shape[1])), g_outputs =np.ones((Y.shape[1])))
DDF_results = model.dea_ddf(X,Y)
columns = ["DDF"]
df = pd.DataFrame(DDF_results, columns=columns)
df['Rank'] = df['DDF'].rank(ascending=True)
display(df)
```
|    |          DDF          | Rank |
|----|-----------------------|------|
| 0  | 1.059028e-01          | 8.0  |
| 1  | 1.302850e-11          | 5.0  |
| 2  | 1.227791e-10          | 7.0  |
| 3  | 2.474448e+00          | 14.0 |
| 4  | 1.122606e-11          | 4.0  |
| 5  | 3.753739e+00          | 15.0 |
| 6  | 2.407980e-01          | 9.0  |
| 7  | 1.781102e+00          | 13.0 |
| 8  | 8.546978e-12          | 1.0  |
| 9  | 1.252857e+00          | 10.0 |
| 10 | 9.724485e-12          | 2.0  |
| 11 | 1.593886e-11          | 6.0  |
| 12 | 1.777659e+00          | 12.0 |
| 13 | 9.978950e-12          | 3.0  |
| 14 | 1.562500e+00          | 11.0 |


### Print the $s$-curves

```python
model.graph_all()
```
![Example Image](CD-order-1.91.png)

