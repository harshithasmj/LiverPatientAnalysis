
# **Overview**

Indian Liver Patient Dataset (ILPD) from the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/). Perform feature engineering,
data preprocessing, dimensionality reduction, and logistic regression to
predict whether a patient has liver disease.
:::

::: {.cell .code execution_count="1" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":616}" id="lGcynrB5Rn1x" outputId="9f933eab-33a4-4034-caeb-48ca464defc3"}
``` python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
pd.set_option('display.max_columns', None)
```

::: {.output .error ename="ImportError" evalue="initialization failed"}
    ---------------------------------------------------------------------------
    KeyboardInterrupt                         Traceback (most recent call last)
    KeyboardInterrupt: 

    The above exception was the direct cause of the following exception:

    ImportError                               Traceback (most recent call last)
    /tmp/ipython-input-1-3038132274.py in <cell line: 0>()
          2 import pandas as pd
          3 import numpy as np
    ----> 4 from sklearn.preprocessing import StandardScaler
          5 import matplotlib.pyplot as plt
          6 from sklearn.linear_model import LogisticRegression

    /usr/local/lib/python3.11/dist-packages/sklearn/__init__.py in <module>
         71     _distributor_init,
         72 )
    ---> 73 from .base import clone  # noqa: E402
         74 from .utils._show_versions import show_versions  # noqa: E402
         75 

    /usr/local/lib/python3.11/dist-packages/sklearn/base.py in <module>
         17 from ._config import config_context, get_config
         18 from .exceptions import InconsistentVersionWarning
    ---> 19 from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr
         20 from .utils._metadata_requests import _MetadataRequester, _routing_enabled
         21 from .utils._param_validation import validate_parameter_constraints

    /usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py in <module>
         13 from . import _joblib, metadata_routing
         14 from ._bunch import Bunch
    ---> 15 from ._chunking import gen_batches, gen_even_slices
         16 from ._estimator_html_repr import estimator_html_repr
         17 

    /usr/local/lib/python3.11/dist-packages/sklearn/utils/_chunking.py in <module>
          9 
         10 from .._config import get_config
    ---> 11 from ._param_validation import Interval, validate_params
         12 
         13 

    /usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in <module>
         15 
         16 from .._config import config_context, get_config
    ---> 17 from .validation import _is_arraylike_not_scalar
         18 
         19 

    /usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in <module>
         19 from .. import get_config as _get_config
         20 from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
    ---> 21 from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
         22 from ..utils.deprecation import _deprecate_force_all_finite
         23 from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype

    /usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in <module>
         15 
         16 from .._config import get_config
    ---> 17 from .fixes import parse_version
         18 
         19 _NUMPY_NAMESPACE_NAMES = {"numpy", "array_api_compat.numpy"}

    /usr/local/lib/python3.11/dist-packages/sklearn/utils/fixes.py in <module>
         15 import scipy
         16 import scipy.sparse.linalg
    ---> 17 import scipy.stats
         18 
         19 try:

    /usr/local/lib/python3.11/dist-packages/scipy/stats/__init__.py in <module>
        622 from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
        623                                DegenerateDataWarning, FitError)
    --> 624 from ._stats_py import *
        625 from ._variation import variation
        626 from .distributions import *

    /usr/local/lib/python3.11/dist-packages/scipy/stats/_stats_py.py in <module>
         39 from scipy.spatial import distance_matrix
         40 
    ---> 41 from scipy.optimize import milp, LinearConstraint
         42 from scipy._lib._util import (check_random_state, _get_nan,
         43                               _rename_parameter, _contains_nan,

    /usr/local/lib/python3.11/dist-packages/scipy/optimize/__init__.py in <module>
        433 from ._nnls import nnls
        434 from ._basinhopping import basinhopping
    --> 435 from ._linprog import linprog, linprog_verbose_callback
        436 from ._lsap import linear_sum_assignment
        437 from ._differentialevolution import differential_evolution

    /usr/local/lib/python3.11/dist-packages/scipy/optimize/_linprog.py in <module>
         19 from ._optimize import OptimizeResult, OptimizeWarning
         20 from warnings import warn
    ---> 21 from ._linprog_highs import _linprog_highs
         22 from ._linprog_ip import _linprog_ip
         23 from ._linprog_simplex import _linprog_simplex

    /usr/local/lib/python3.11/dist-packages/scipy/optimize/_linprog_highs.py in <module>
         18 from ._optimize import OptimizeWarning, OptimizeResult
         19 from warnings import warn
    ---> 20 from ._highspy._highs_wrapper import _highs_wrapper
         21 from ._highspy._core import(
         22     kHighsInf,

    /usr/local/lib/python3.11/dist-packages/scipy/optimize/_highspy/_highs_wrapper.py in <module>
          2 
          3 import numpy as np
    ----> 4 import scipy.optimize._highspy._core as _h # type: ignore[import-not-found]
          5 from scipy.optimize._highspy import _highs_options as hopt  # type: ignore[attr-defined]
          6 from scipy.optimize import OptimizeWarning

    ImportError: initialization failed

    ---------------------------------------------------------------------------
    NOTE: If your import is failing due to a missing package, you can
    manually install dependencies using either !pip or !apt.

    To view examples of installing some common dependencies, click the
    "Open Examples" button below.
    ---------------------------------------------------------------------------
:::
:::

::: {.cell .markdown id="rECF3w0hRn1v"}
## **Description of the Dataset**

-   The ILPD comprises 583 patient records with 10
    biochemical/demographic features and a target feature that are
    listed below:

The dataset has the following columns:

-   `Age` (Integer): Patient\'s age.
-   `Gender` (Categorical: Male/Female).
-   `TB` (Total Bilirubin, Continuous).
-   `DB` (Direct Bilirubin, Continuous).
-   `Alkphos` (Alkaline Phosphotase, Integer).
-   `Sgpt` (Alamine Aminotransferase, Integer).
-   `Sgot` (Aspartate Aminotransferase, Integer).
-   `TP` (Total Proteins, Continuous).
-   `ALB` (Albumin, Continuous).
-   `A/G Ratio` (Albumin and Globulin Ratio, Continuous).
-   `Selector` (Binary: 1 = Liver patient, 2 = Non-liver patient)
:::

::: {.cell .markdown id="3XHDknmzRn1v"}
### **Import libraries, Load the Dataset, and Create a Header**
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":206}" id="SrqFK9NTZEps" outputId="8696c42f-8635-4a69-e8b3-94747f3979b7"}
``` python
column_names = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio', 'Selector']
df = pd.read_csv("ILPD.csv", names = column_names)
df['Target'] = df['Selector'].apply(lambda x:1 if x == 1 else 0)

df = df.drop(columns=['Selector'])

column_namesnew = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio', 'Target']
df.columns = column_namesnew
df.head(5)
```

::: {.output .execute_result execution_count="26"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 583,\n  \"fields\": [\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 16,\n        \"min\": 4,\n        \"max\": 90,\n        \"num_unique_values\": 72,\n        \"samples\": [\n          46,\n          23,\n          63\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"Male\",\n          \"Female\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"TB\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 6.209521726180145,\n        \"min\": 0.4,\n        \"max\": 75.0,\n        \"num_unique_values\": 113,\n        \"samples\": [\n          4.9,\n          3.9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DB\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.8084976176589636,\n        \"min\": 0.1,\n        \"max\": 19.7,\n        \"num_unique_values\": 80,\n        \"samples\": [\n          6.2,\n          0.1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Alkphos\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 242,\n        \"min\": 63,\n        \"max\": 2110,\n        \"num_unique_values\": 263,\n        \"samples\": [\n          386,\n          209\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sgpt\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 182,\n        \"min\": 10,\n        \"max\": 2000,\n        \"num_unique_values\": 152,\n        \"samples\": [\n          2000,\n          321\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sgot\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 288,\n        \"min\": 10,\n        \"max\": 4929,\n        \"num_unique_values\": 177,\n        \"samples\": [\n          66,\n          16\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"TP\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.0854514840234664,\n        \"min\": 2.7,\n        \"max\": 9.6,\n        \"num_unique_values\": 58,\n        \"samples\": [\n          6.8,\n          6.7\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ALB\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.795518805964027,\n        \"min\": 0.9,\n        \"max\": 5.5,\n        \"num_unique_values\": 40,\n        \"samples\": [\n          2.0,\n          1.9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"A/G Ratio\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.31959210767237095,\n        \"min\": 0.3,\n        \"max\": 2.8,\n        \"num_unique_values\": 69,\n        \"samples\": [\n          1.6,\n          0.9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Target\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .markdown id="VTp2JazQRn1x"}
### **Identify columns with missing data (if any) and then determine an appropriate strategy for each**
:::

::: {.cell .code id="fbMnaWRFRn1y"}
``` python
before_NA = df.isnull().sum()

numerical = ['Age', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio']
df[numerical] = df[numerical].fillna(df[numerical].median())


df['Gender'].fillna(df['Gender'].mode())
df['Gender'].fillna(df['Gender'].mode())

after_NA = df.isnull().sum()
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="wILfx-JHcv13" outputId="0cd53527-aaab-4e90-efe0-91f102d0c04d"}
``` python


assert after_NA.sum() == 0, "Check for missing values."


print("No missing values.")



```

::: {.output .stream .stdout}
    No missing values.
:::
:::

::: {.cell .markdown id="QJhpak0lRn1y"}
### **One-hot Encoding for Categorical Variables**
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":478}" id="JKqxUcSQRn1z" outputId="5d094fef-44e5-4228-f335-6d0001029951"}
``` python

df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
df
```

::: {.output .stream .stderr}
    <ipython-input-29-aabfa18471b2>:3: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
:::

::: {.output .execute_result execution_count="29"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 583,\n  \"fields\": [\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 16,\n        \"min\": 4,\n        \"max\": 90,\n        \"num_unique_values\": 72,\n        \"samples\": [\n          46,\n          23,\n          63\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"TB\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 6.209521726180145,\n        \"min\": 0.4,\n        \"max\": 75.0,\n        \"num_unique_values\": 113,\n        \"samples\": [\n          4.9,\n          3.9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DB\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.8084976176589636,\n        \"min\": 0.1,\n        \"max\": 19.7,\n        \"num_unique_values\": 80,\n        \"samples\": [\n          6.2,\n          0.1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Alkphos\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 242,\n        \"min\": 63,\n        \"max\": 2110,\n        \"num_unique_values\": 263,\n        \"samples\": [\n          386,\n          209\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sgpt\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 182,\n        \"min\": 10,\n        \"max\": 2000,\n        \"num_unique_values\": 152,\n        \"samples\": [\n          2000,\n          321\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sgot\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 288,\n        \"min\": 10,\n        \"max\": 4929,\n        \"num_unique_values\": 177,\n        \"samples\": [\n          66,\n          16\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"TP\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.0854514840234664,\n        \"min\": 2.7,\n        \"max\": 9.6,\n        \"num_unique_values\": 58,\n        \"samples\": [\n          6.8,\n          6.7\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ALB\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.795518805964027,\n        \"min\": 0.9,\n        \"max\": 5.5,\n        \"num_unique_values\": 40,\n        \"samples\": [\n          2.0,\n          1.9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"A/G Ratio\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.3184950795922123,\n        \"min\": 0.3,\n        \"max\": 2.8,\n        \"num_unique_values\": 69,\n        \"samples\": [\n          1.6,\n          0.9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Target\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .markdown id="27I-GbBmRn10"}
### **Feature Extraction**
:::

::: {.cell .code id="pzMuXEFZRn10"}
``` python
df['Bilirubin Ratio'] = df['DB'] / df['TB']
df['ALT/AST Ratio'] = df['Sgpt'] / df['Sgot']
```
:::

::: {.cell .markdown id="RPnDFfsZRn1z"}
### **Standardization**
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":244}" id="6m5vP0kxRn10" outputId="99488ec0-f930-4292-a71f-9e3a4028a9c1"}
``` python

scaler = StandardScaler()
df[numerical] = scaler.fit_transform(df[numerical])
df.head()
```

::: {.output .execute_result execution_count="69"}
``` json
{"type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .markdown id="AOE9q27PRn11"}
### **Engineering Ordinal Features**
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":244}" id="mH6jp0IZduTi" outputId="60265392-b8d4-4794-d772-961f9e558002"}
``` python
age_min = df['Age'].min()
age_max = df['Age'].max()


age_bins = sorted(set([age_min, 30, 50, age_max]))
age_labels = ['Young', 'Middle-Aged', 'Senior']


if len(age_bins) < 3:
    age_bins = [age_min, (age_min + age_max) / 2, age_max]
    age_labels = ['Young', 'Senior']


df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels = age_labels, include_lowest = True)
df['age_ord'] = df['age_group'].apply(lambda x:1 if x == 'Young' else (2 if x == 'Middle-Aged' else 3))
df['age_ord'].fillna(df['age_ord'].mode())
df['Age'].fillna(df['Age'].median())
# Map ordinal categories
# Display first few rows
df.head()
```

::: {.output .execute_result execution_count="96"}
``` json
{"type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .markdown id="RcDFR89WRn10"}
### **Dimensionality Reduction**
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="nQ9IzJZiRn10" outputId="28d0d934-2fd9-46c8-b14a-090d1fd22741"}
``` python


num_df = df.drop(columns=['Target', 'Gender', 'age_group', 'age_ord'])
X = num_df
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_before = model.predict(X_test)
accuracy_bef= accuracy_score(y_test, y_pred_before)
print("Accuracy before:", accuracy_bef)

scaler = StandardScaler()
scaled = scaler.fit_transform(X)

pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model.fit(X_train_pca, y_train)
y_pred_after_4 = model.predict(X_test_pca)
accuracy_after_4 = accuracy_score(y_test, y_pred_after_4)
print("Accuracy after:", accuracy_after_4)

pca = PCA(n_components=2)
X_train_pca_2 = pca.fit_transform(X_train)
X_test_pca_2 = pca.transform(X_test)

model.fit(X_train_pca_2, y_train)
y_pred_after_2 = model.predict(X_test_pca_2)
accuracy_after_2 = accuracy_score(y_test, y_pred_after_2)
print("Accuracy after:",accuracy_after_2)

```

::: {.output .stream .stdout}
    Accuracy before: 0.7521367521367521
    Accuracy after: 0.717948717948718
    Accuracy after: 0.717948717948718
:::
:::

::: {.cell .markdown id="76bqrlyFVBHr"}
## **Logistic Regression Model**

-   Train a logistic regression model using the processed dataset.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="awfkBZULVHDt" outputId="3afcf95a-76e4-4e4c-8fcf-177311129327"}
``` python
df = pd.get_dummies(df, columns=['age_group'])

X = df.drop(columns=['Target'])
y1 = df['Target']

X_train, X_test, y_train, y = train_test_split(X, y1, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

heatmap = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(heatmap)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()
# Evaluate model
print("Accuracy:", accuracy_score(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred))


#coeffients
feature = X.columns
coefficient = model.coef_[0]
coefdf = pd.DataFrame({'Feature': feature, 'Coefficient': coefficient})
coefdf['Abs_Coefficient'] =  coefdf['Coefficient'].abs()
coefdf = coefdf.sort_values(by='Abs_Coefficient')


plt.figure(figsize=(10,6))
sns.barplot(x='Abs_Coefficient', y='Feature', data = coefdf)
plt.title("Features Based on Logistic Regression Coefficients")
plt.xlabel("Absolute Coefficient Value")
plt.ylabel("Feature")
plt.show()
```

::: {.output .display_data}
![](68f3c5a8aa53863ed98757bb2773face7ddb37f0.png)
:::

::: {.output .stream .stdout}
    Accuracy: 0.7606837606837606
    Confusion Matrix:
     [[ 9 21]
     [ 7 80]]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.56      0.30      0.39        30
               1       0.79      0.92      0.85        87

        accuracy                           0.76       117
       macro avg       0.68      0.61      0.62       117
    weighted avg       0.73      0.76      0.73       117
:::

::: {.output .display_data}
![](70ff1d54c24fe29d2def578433411942e50d24ff.png)
:::
:::

::: {.cell .markdown id="ZLvb_Be_jAyB"}
## **Visualisation**

-   Plot the distribution of Total Bilirubin (TB) for liver and
    non-liver patients.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":554}" id="9iYEVkOjjIuH" outputId="d6ffeaeb-8311-4912-818a-a499ca7ecfbc"}
``` python
import matplotlib.pyplot as plt
import seaborn as sns

liver_patients = df[df['Target'] == 1]
non_liver_patients = df[df['Target'] == 0]
plt.figure(figsize=(10, 6))
plt.hist(liver_patients['TB'], color='red', label='Liver Patients', alpha = 0.5)
plt.hist(non_liver_patients['TB'], color='blue', label='Non-Liver Patients')
plt.xlabel('Total Bilirubin (TB)')
plt.ylabel('Frequency')
plt.title('Total Bilirubin (TB) for liver and non-liver patient')
plt.legend()
plt.show()
```

::: {.output .display_data}
![](060251cdbd9a700e16fabcaafb8f945f77ef78da.png)
:::
:::

::: {.cell .markdown id="ITiyWr1FVyv5"}
**Based on the confusion matrix and classification report, this model
perform pretty well to identifying liver patients based on the TB. As
the F1 score is high as 0.85, this model is precise for Liver or
Class 1. The recall for non-liver patients is at 0.30 indicating that it
fails to correctly diagnore non-liver patients (about 21 false
positives). The model is incorrectly classifies non-liver patients as
liver patients and increases the number of false positives that may be a
problem in diagnosis. But, false negatives are more concerning as
failing to diagnose a patient has more severe consequences than wrongly
diagnosing that a patient has liver disease.**
:::

::: {.cell .markdown id="cD-L_JkyWCOs"}
**TP, DB, Sgot, Sgpt, and ALB are the features that seem to have the
strongest impact on the predictions pf liver disearse based on the
logistic regression coefficients. These findings make sense because the
five features are the main trackers to ensure one\'s liver health is in
good condition. TP, DB, Sgot, Sgpt, and ALB help assess liver health.
FOr example, SGOT and SGPT are two common enzymes produced by the liver.
TP indicates the total protein.**
:::

::: {.cell .markdown id="koWRdT3CdG1r"}
**These transformations to create engineered features can help the model
by observing important relationship to increase accuracy of diagonsis
for liver disease. But, this may result in bias in the predictions due
to improper binning and not being properly scaled. This may cause the
new variables to not have accurate predictions based on the data.**
:::

::: {.cell .markdown id="yOyaxN1LWf6u"}
**Both distributions are right tailed with higher frequency data points
for liver patients. This graph may indicate that those with higher TB
are more likely to be liver patients. There is a noticeable difference
between liver and non-liver patients (red and blue). Higher values of TB
are associated with liver disease patients for most of the data
distribution. Therefore, TB is a strong predictor for detecting liver
disease.**
:::

::: {.cell .code id="O8OllkGbyfpL"}
``` python
```
:::
