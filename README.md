# reparo

Reparo is a python sci-kit learn inspired package for Missing Value Imputation. It contains a some feature transformers to eliminate Missing Values (NaNs) from your data for Machine Learning Algorithms.

This version of reparo has the next methods of missing value imputation:
1) Cold-Deck Imputation (CDI).
2) Hot-Deck Imputation (HotDeckImputation).
3) Fuzzy-Rough Nearest Neighbor for Imputation (FRNNI).
4) K-Nearest Neighbors Imputation (KNNI).
5) Single Center Imputation from Multiple Chained Equation (SICE).
6) Predictive Mean Matching (PMM).
7) Multivariate Imputation by Chained Equation (MICE).

All these methods work like normal sklearn transformers. They have fit, transform and fit_transform functions implemented.

Additionally every reparo transformer has an apply function which allows to apply an transformation on a pandas Data Frame.

# How to use reparo
To use a transformer from reparo you should just import the transformer from reparo in the following framework:

```python
from reparo import MICE
```

class names are written above in parantheses.

Next create a object of this algorithm (I will use k-Nearest Neighbors Imputation as an example).

```python
method = KNNI()
```

Firstly you should fit the transformer, passing to it a feature matrix (X) and the target array (y). y argument is not really used (as it causes data leackage)

```python
method.fit(X, y)
```

After you fit the model, you can use it for transforming new data, using the transform function. To transform function you should pass only the feature matrix (X).

```python
X_transformed = method.transform(X)
```

Also you can fit and transform the data at the same time using the fit_transform function.

```python
X_transformed = method.fit_transform(X)
```

Also you can apply a transformation directly on a pandas DataFrame, choosing the columns that you want to change.

```python
new_df = method.apply(df, 'target', ['col1', 'col2'])
```

With <3 from Sigmoid.
We are open for feedback. Please send your impression to vladimir.stojoc@gmail.com
