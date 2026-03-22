# Bias Mitigation in Healthcare with Fairlearn

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Fairlearn](https://img.shields.io/badge/Fairlearn-6C3483?style=flat&logoColor=white)

> **TL;DR:** Mitigating racial bias in glioblastoma tumor detection using XGBoost and Fairlearn postprocessing. After applying ThresholdOptimizer, minority recall improved from **0.82 → 0.91** while maintaining overall model performance.

| | Recall Before | Recall After |
|---|---|---|
| **Minority** | 0.818 | **0.909** |
| **White** | 0.926 | 0.912 |

---

## Fairness in Healthcare

Many of the healthcare outcomes we try to predict are rare (sometimes less than .1% of observations), and even more rare in minority groups, due to issues ranging from social determinants of health, healthcare disparities, or lack of data. Because of this we shift focus on reducing harm and in this case want to reduce the number of false negatives or misdiagnosis in our models.

This is more true in the American healthcare system where healthcare resources are strained, so healthcare overall is triaging and not expanding fast enough to accommodate growing needs. Models such as those below are tools used to help triage and prioritize growing needs, increasing their influence in decision making within healthcare.

The types of harm I will focus on are allocation harms and quality of service harms. Underdiagnosing could potentially lead to worse healthcare outcomes and resources not provided by programs such as care management. We see these disparities across our sensitive feature of interest.

## Data

The data we use for this example is based on the Glioma Grading dataset provided by the National Cancer Institute (NCI) and found in UC Irvine's Machine Learning Repository. This classification dataset determines whether a patient tumor is lower grade glioma (0) or glioblastoma multiforme (1) or aggressive form of tumor found in the brain and is supported by 23 features primarily focused on mutations. The sensitive feature we will use from this dataset is the Race category. 

Data can be easily accessed via the ucimlrepo.

```python
from ucimlrepo import fetch_ucirepo

glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759) 
```

With too few minority observations I have decided to combine them. This highlights many of the common data issues, or lack of data we see within minority groups. 

![Race distribution](/images/race_volumes.png)

## Model

One of my favorite packages which I have the most experience with is XGBoost. I will be using the scikit-learn estimator interface in combination with a data prep component using a pipeline. This is my current favorite version. Other versions to test include the base XGBoost, PySpark API, Dask, Ray, R, a recently revamped Sparklyr versions, some with their own nuances, but this and the base version are my favorites so far. Before assigning our predictor to our machine learning pipeline I do some light hyperparameter tuning with our focus for this repo primarily targeting fairness through recall.

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

tuning_pipeline = Pipeline([
    ("prep_data", preprocessing),
    ("xgboost", xgb.XGBClassifier(objective= "binary:logistic"
                                  , tree_method = "hist"
                                  , random_state = 42))
])

param_grid = [
    {'xgboost__learning_rate': [.01, .1, .3, .5],
     'xgboost__max_depth': [4, 6, 8, 10]}
]
```


## Fairness Assessments: Disaggregated Metrics

Fairlearn can be divided into two primary components, the assessment portion focuses on identifying potential harm by identifying gaps or differences in model metrics across different sensitive features. The subsequent component mitigates the observed differences across sensitive features.

This tool automates a process we had formerly done across different groups ranging from gender, age groups, race, and many more. This is great for quickly identifying large gaps between groups before and post mitigation.

This approach creates a 'MetricFrame' of all metrics overall and across different groups. For our example we observe race. Fairlearn includes several metrics including selection rate. For our analysis we use sklearn's classification metrics.

```python
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count
from sklearn.metrics import recall_score, accuracy_score, balanced_accuracy_score

my_metrics = {
    'recall' : recall_score,
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'count' : count
}

mf = MetricFrame(
    metrics = my_metrics,
    y_true = y_test,
    y_pred = preds, 
    sensitive_features = X_test['Race']
)

mf.overall
```

Accuracy is not great, but also not bad. Recall overall looks very promising. This means we are able to accurately capture 91% of all actual positive cases.

```
recall                 0.911392
accuracy               0.857143
balanced_accuracy      0.860191
count                168.000000
dtype: float64
```

We can leverage the available methods within our MetricFrame and now observe the recall across race. We see a substantial difference in recall between our minority class. This gap means we are more likely to underdiagnose actual minority groups with a severe tumor. This includes various types of harm such as harm of allocation, because we do not identify a minority patient with glioblastoma multiforme (1) they are unlikely to be treated. This also encapsulates a different harm, defined as harm of quality of service, a minority patient is not receiving the same level of diagnosis of glioblastoma multiforme as their counterpart.

```python
mf.by_group
```

<center>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recall</th>
      <th>accuracy</th>
      <th>balanced_accuracy</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Race</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>minority</th>
      <td>0.818182</td>
      <td>0.722222</td>
      <td>0.694805</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>white</th>
      <td>0.926471</td>
      <td>0.873333</td>
      <td>0.877869</td>
      <td>150.0</td>
    </tr>
  </tbody>
</table>
</div>

</center>

This example is very straightforward but sometimes when you have many sensitive features you need to test you need to quickly identify the largest gaps. We can leverage an additional method 

```python
mf.difference(method='between_groups')
```

Because we are only comparing two classes within a single sensitive feature, we see the differences in recall are equal to .108289, or the difference in recall values above.

```
recall                 0.108289
accuracy               0.151111
balanced_accuracy      0.183064
count                132.000000
dtype: float64
```


![Disaggregated metrics](/images/all_metrics.png)


## Fairness Mitigation: Postprocessing with Thresholding

The second component addresses the differences noted above. They include a suite of solutions, however we will focus on using the postprocessing approach. Postprocessing means this is addressed following training of the model. This method is currently limited to classification problems with two potential outcomes. 

The function used is the _ThresholdOptimizer()_. This comprises of several arguments from our estimator, which can include a pipeline or a simple estimator, constraints which is our metric of interest we are trying to balance amongst our classes, and objective or overall metric to assess if overall model results balanced. 

```python
from fairlearn.postprocessing import ThresholdOptimizer

threshold_optimizer = ThresholdOptimizer(
    estimator=xgb_pipeline,
    constraints="true_positive_rate_parity", # recall
    objective="accuracy_score", 
    predict_method="predict_proba", 
    prefit=False,
)

threshold_optimizer.fit(X_train, y_train, sensitive_features=X_train['Race'])

```

This fits our training data and tests several thresholds (XGBoost default = .5) that determine assignment to positive or negative cases based on our model scores or probabilities. The end result is a threshold selected that maximizes both objective and constraints. 

```python
from fairlearn.postprocessing import plot_threshold_optimizer

plot_threshold_optimizer(threshold_optimizer)
```
![Threshold optimizer](/images/threshold_optimizer_plot.png)

Once fitted we can generate new predictions with our new thresholds.

```python
thresh_preds = threshold_optimizer.predict(X_test, sensitive_features=X_test['Race'], random_state=12345)

```

Let's compare before and after results. We see a significant increase in recall for our minority class.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recall before mitigation</th>
      <th>recall after mitigation</th>
    </tr>
    <tr>
      <th>Race</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>minority</th>
      <td>0.818182</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>white</th>
      <td>0.926471</td>
      <td>0.911765</td>
    </tr>
  </tbody>
</table>
</div>

<br>

Though this model helps identify potential health issues, this model should only be used as a recommendation tool and not a decision maker. A human component must be included to capture context not identified in the data, such as a patient's decision to forgo treatment due to quality of life preferences. 

## Project Structure

```
├── README.md
├── .gitignore
├── requirements.txt
├── data/
├── images/
├── models/
│   ├── my_model.keras
│   └── xgb.pkl
└── notebooks/
    ├── exploratory_data_analysis.ipynb
    ├── xgboost_fairlearn.ipynb
    └── experimental/              # Work-in-progress explorations
```

## Appendix

### Citations

Taski, Erdal, Camphausen, Kevin, Krauze, Andra Valentina, and Zhuge, Ying. (2022). Glioma Grading Clinical and Mutation Features. UCI Machine Learning Repository. https://doi.org/10.24432/C5R62J.

### Packages

altair — I enjoy creating visualizations using this package as it drives the idea of building on layers, e.g. aesthetics, facets. Recommend reading below.

 - https://byrneslab.net/classes/biol607/readings/wickham_layered-grammar.pdf
