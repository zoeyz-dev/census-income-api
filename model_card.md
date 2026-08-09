# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a `RandomForestClassifier` from scikit-learn, trained with 100 estimators,
max depth of 16, and `random_state=42`. It was developed as part of the Udacity Machine
Learning DevOps Engineer Nanodegree capstone project. Categorical features are one-hot
encoded with `OneHotEncoder(handle_unknown="ignore")`, and the label is binarized with
`LabelBinarizer`. The trained model, encoder, and label binarizer are serialized with
`joblib` and saved under `model/`.

## Intended Use

The model predicts whether a person's annual income exceeds $50,000 based on demographic
and employment attributes from US Census data. It is intended for a classroom exercise in
building and deploying a machine learning pipeline (training, testing on data slices, and
serving predictions through a REST API), not for real-world income determination, credit
decisions, or any use affecting an individual's employment, benefits, or financial status.

## Training Data

The data is the UCI Census Income (Adult) dataset, containing 32,561 rows of US Census
records with 14 features (age, workclass, education, marital status, occupation,
relationship, race, sex, native country, and several continuous features) plus a binary
salary label (`<=50K` or `>50K`). The dataset was split 80/20 into training and test sets
with `train_test_split(test_size=0.20, random_state=42)`. The eight categorical features
(workclass, education, marital-status, occupation, relationship, race, sex,
native-country) are one-hot encoded; the label is binarized to 0/1.

## Evaluation Data

The evaluation data is the 20% test split held out from the same Adult dataset, processed
with the encoder and label binarizer fit on the training data (no separate fit on test
data). Overall metrics and per-slice metrics on this test set are written to
`slice_output.txt` each time `train_model.py` runs.

## Metrics

The model is evaluated with precision, recall, and F1 (fbeta with beta=1). On the held-out
test set:

- Precision: 0.7944
- Recall: 0.5805
- Fbeta: 0.6708

These numbers come from a fresh run of `train_model.py` on the current codebase. Precision
is higher than recall, meaning the model is more likely to miss a true `>50K` case than to
wrongly flag a `<=50K` case as `>50K`.

`compute_model_metrics` uses `zero_division=1`, so a slice with no actual `>50K` cases
reports recall and fbeta of 1.0000 by convention rather than by measurement, and a slice
where the model predicts no `>50K` cases reports precision 1.0000 the same way.
`slice_output.txt` prints the actual-positive count (`n_pos`) and predicted-positive count
(`n_pred_pos`) for every slice so these rows can be told apart from a real result. Of the
99 slice rows in `slice_output.txt`, 21 report 1.0000 on all three metrics because
`n_pos=0` and `n_pred_pos=0` together (for example `Without-pay`, n=4; `1st-4th`, n=23;
`Preschool`, n=10), and 42 report precision 1.0000 because `n_pred_pos=0` alone.

## Ethical Considerations

The training data includes race, sex, and native country as input features. Because the
model conditions on these attributes, its predictions can reproduce demographic disparities
already present in the 1994 Census data it was trained on, which reflects historical labor
market patterns rather than current conditions. The per-slice metrics in
`slice_output.txt` show the model does not perform uniformly across categories, so any use
beyond this training exercise would need a fairness audit before predictions are applied to
real people.

## Caveats and Recommendations

Slice performance in `slice_output.txt` varies with both feature value and sample size. For
`workclass = Private` (n=4578, the largest slice), performance is close to the overall
average: precision 0.8120, recall 0.5564, fbeta 0.6603. For `workclass = Without-pay`
(n=4, n_pos=0), the reported metrics are precision 1.0000, recall 1.0000, fbeta 1.0000; this
is the `zero_division=1` convention described under Metrics, not a real perfect result, and
should not be trusted as a performance estimate for that group.

A pattern that is a real result, not a convention artifact: recall collapses on
low-education slices while precision stays high, meaning the model consistently
under-predicts `>50K` for these groups rather than over-predicting it. `10th`
(n=183, n_pos=12) has recall 0.0833; `7th-8th` (n=141, n_pos=6) has recall 0.0000;
`Own-child` (n=1019, n_pos=17) has recall 0.1765. Each of these has enough actual `>50K`
cases (n_pos >= 6) that the low recall reflects the model's behavior on that group rather
than an empty-slice artifact. Any deployment decision should weight slices by sample size,
distinguish zero-division rows from measured ones using `n_pos`/`n_pred_pos`, and re-evaluate
on more balanced or more recent data before drawing conclusions about small or rare
categories.
