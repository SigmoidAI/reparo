from distutils.core import setup
long_description = '''
Reparo is a python sci-kit learn inspired package for Missing Value Imputation. It contains a some feature transformers to eliminate Missing Values (NaNs) from your data for Machine Learning Algorithms.\n
This version of reparo has the next methods of missing value imputation:\n
1) Cold-Deck Imputation (CDI).\n
2) Hot-Deck Imputation (HotDeckImputation).\n
3) Fuzzy-Rough Nearest Neighbor for Imputation (FRNNI).\n
4) K-Nearest Neighbors Imputation (KNNI).\n
5) Single Center Imputation from Multiple Chained Equation (SICE).\n
6) Predictive Mean Matching (PMM).\n
7) Multivariate Imputation by Chained Equation (MICE).\n
All these methods work like normal sklearn transformers. They have fit, transform and fit_transform functions implemented.\n
Additionally every reparo transformer has an apply function which allows to apply an transformation on a pandas Data Frame.\n
How to use reparo\n
To use a transformer from reparo you should just import the transformer from reparo in the following framework:\n
```from reparo import <class name>```\n
class names are written above in parantheses.\n
Next create a object of this algorithm (I will use k-Nearest Neighbors Imputation as an example).\n
```method = KNNI()```\n
Firstly you should fit the transformer, passing to it a feature matrix (X) and the target array (y). y argument is not really used (as it causes data leackage)\n
```method.fit(X, y)```\n
After you fit the model, you can use it for transforming new data, using the transform function. To transform function you should pass only the feature matrix (X).\n
```X_transformed = method.transform(X)```\n
Also you can fit and transform the data at the same time using the fit_transform function.\n
```X_transformed = method.fit_transform(X)```\n
Also you can apply a transformation directly on a pandas DataFrame, choosing the columns that you want to change.\n
```new_df = method.apply(df, 'target', ['col1', 'col2'])```\n
With love from Sigmoid.\n
We are open for feedback. Please send your impression to papaluta.vasile@isa.utm.md\n
'''

setup(
  name = 'reparo',
  packages = ['reparo'],
  version = '0.1.0',
  license='MIT',
  description = 'Reparo is a python sci-kit learn inspired package for Missing Value Imputation.',
  long_description=long_description,
  author = 'SigmoidAI - Stojoc Vladimir',
  author_email = 'vladimir.stojoc@gmail.com',
  url = 'https://github.com/SigmoidAI/reparo',
  download_url = 'https://github.com/ScienceKot/kydavra/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['ml', 'machine learning', 'feature engineering', 'python', 'data science'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'scikit-learn',
          'statsmodels'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Framework :: Jupyter',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
  ],
    long_description_content_type='text/x-rst',
)