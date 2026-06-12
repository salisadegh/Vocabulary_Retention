# Measurement lenses for L2 vocabulary retention

Code and data for a two-study comparison of behavioural learning traces and
self-reported strategy use as predictors of second-language vocabulary
retention.

## Contents

```
data/
  duolingo_raw/        (user-generated) original public dataset
  duolingo_sample/     (user-generated) deterministic 15% user-complete sample
study1_duolingo/
  download_data.sh     fetch the public dataset from Harvard Dataverse
  make_sample.py       deterministic sampler (md5(user_id) % 100 < 15)
  manifest_reference.txt  MD5 manifest of the canonical analysis sample
  study1_trace_level.py   trace-level models under user-level GroupKFold
  study1b_person_level.py person-level prospective analysis (week 1 -> week 2)
  results/             outputs of both analyses (tables, figures, provenance)
study2_field/
  data/study2_data.csv anonymised field-sample data (n = 109)
  study2_analysis.py   pre-defined multi-specification analysis
  study2_supplementary.py  bootstrap CIs, disattenuation, item-level FDR,
                           cross-study common-metric table
  outputs/, figures/   outputs of both scripts
```

## Reproducing the analyses

Run everything from the repository root.

```
pip install -r requirements.txt
bash study1_duolingo/download_data.sh        # ~361 MB from Dataverse
python study1_duolingo/make_sample.py        # rebuilds the exact sample
python study1_duolingo/study1_trace_level.py
python study1_duolingo/study1b_person_level.py
python study2_field/study2_analysis.py
python study2_field/study2_supplementary.py
```

The sampler is deterministic and user-complete: a learner is included iff
`md5(user_id) % 100 < 15`, and included learners contribute all their
traces. The rebuilt sample can be verified byte-for-byte against
`study1_duolingo/manifest_reference.txt` (1,930,889 rows; 17,230 users).
Trace-level cross-validation is grouped by learner, so no learner appears
in both training and test data; half-life-regression hyperparameters are
selected on an inner user-level holdout within each training fold only.

## Data

* **Duolingo learning traces** are not redistributed here. They are publicly
  available from Harvard Dataverse (doi:10.7910/DVN/N8XJME) under the terms
  stated in the dataset record (Settles & Meeder, 2016, ACL), and the exact
  analysis sample is reconstructed deterministically by `make_sample.py`.
* **Field-sample data** (`study2_field/data/study2_data.csv`) are anonymous:
  a sequential participant index, gender, age, two vocabulary test scores,
  three derived retention indicators, and 45 Likert-scale item responses.
  No direct or indirect identifiers are included.

## Licence

Code is released under the MIT License (see LICENSE). The field-sample data
are released for research use with attribution. The Duolingo dataset remains
governed by its own terms on Harvard Dataverse.
