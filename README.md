# 1. BEHRT benchmark

This repository is a fork of [BEHRT](https://github.com/deepmedicine/BEHRT). This package was created for benchmarking
BEHRT on survival tasks.

It is modified to:
- be compatible with [FastEHR](https://github.com/cwlgadd/FastEHR), an ETL package for Electronic Health Records,
- support survival downstream tasks using BEHRT.

There is a difference in the training objectives of survival clinical foundation models, such as SurvivEHR 
(a decoder-only model trained directly on the time-to-event survival task) and BEHRT (an encoder architecture 
with a classification objective). This makes direct comparison very difficult. We overcome this by attaching a 
new survival head to the BEHRT backbone using [DeSurv](https://github.com/djdanks/DeSurv). During our experiments 
we allow fine-tuning to adapt the BEHRT backbone's weights. Within experiments we found that performance of BEHRT
was even worse when used entirely as an encoder for a downstream survival task. Consequently, this package provides
a new, bespoke, model for survival clinical foundation models using the BEHRT encoder architecture.

---

# 2. Results

## Local example

A simple simulated dataset serving as a demonstration of this repository. We simulate 10000 training patients 
(with 100 validation and 100 test) with two visits composed of a number of medical codes. We simulate
a target outcome: for patients who have experienced the medical code I10 the outcome will be death, with a 
probability dependent on the number of codes previously seen. Otherwise the outcome will belong to a set of 
possible other medical codes (which will be treated as right-censored).

To create the ```local_example``` dataset run

```
>> python <repository_path>/task/data/make_local_example_data.py
```

**Pre-training results** 

Masked Language Modelling (MLM) pre-training objective

| Script                    | Metric                              | Result   |
|---------------------------|-------------------------------------|----------|
| `MLM-CPRD.ipynb`          | - Precision                         | **0.5366** |


**Fine-tuning prognostic risk results**

Survival fine-tuning objective

| Scheme           | Outcome                   | Task/Metric                         | Result   |
|------------------|---------------------------|-------------------------------------|----------|
| From scratch     | DEATH                     | Single-risk fine-tuning             |          |
|                  |                           | - Concordance (time-dependent)      | **0.8654** |
|                  |                           | - Integrated Brier Score            | **0.0499** |
|                  |                           | - Negative Bernoulli log-likelihood | **0.1761** | 
|                  |                           |                                     |          | 
| Semi pre-trained | DEATH                     | Single-risk fine-tuning             |          |
|                  |                           | - Concordance (time-dependent)      | **0.8712** |
|                  |                           | - Integrated Brier Score            | **0.0498** |
|                  |                           | - Negative Bernoulli log-likelihood | **0.1758** | 

---

## FastEHR example

This is the filler dataset that is provided by the FastEHR package to demonstrate it's utility. As this is purely
demonstrative there is no signal to model.

---

## SurvivEHR benchmarks

We benchmark on the same tasks as [SurvivEHR](https://www.medrxiv.org/content/medrxiv/early/2025/08/06/2025.08.04.25332916.full.pdf "medrxiv preprint").
SurvivEHR was trained and evaluated using **Clinical Practice Research Datalink (CPRD)** derived datasets. Here we benchmark BEHRT on these datasets.
Note, these datasets are not publically available.


**Pre-training results** 

Masked Language Modelling (MLM) pre-training objective


| Population                 | Script           | Metric                              | Result   |
|----------------------------|------------------|-------------------------------------|----------|
| Type 2 Diabetes Mellitus   | `MLM-CPRD.ipynb` | - Precision                         | **0.6283** | 
|                            |                  |                                     |          |
| Multi-morbidity            | `MLM-CPRD.ipynb` | - Precision                         | *(pending)* |

**Fine-tuning prognostic risk results**

Survival fine-tuning objective

| Population                 | Outcome                  | Scheme                    | Task/Metric                         | Result   |
|----------------------------|--------------------------|---------------------------|-------------------------------------|----------|
| Type 2 Diabetes Mellitus   | Hypertension             | From scratch              | Single-risk fine-tuning             |          |
|                            |                          |                           | - Concordance (time-dependent)      | **0.7256** |
|                            |                          |                           | - Integrated Brier Score            | **0.0879** |
|                            |                          |                           | - Negative Bernouli log-likelihood  | **0.2982** |
|                            |                          |                           |                                     |          |
|                            |                          | Semi pre-trained          | Single-risk fine-tuning             |          |
|                            |                          |                           | - Concordance (time-dependent)      | **0.7297** |
|                            |                          |                           | - Integrated Brier Score            | **0.0870** |
|                            |                          |                           | - Negative Bernouli log-likelihood  | **0.2946** |
|                            |                          |                           |                                     |          |
|                            | Cardiovascular Disease   | From scratch              | Competing-risk fine-tuning          |          |
|                            | - 5 CVD medical ICD codes|                           | - Concordance (time-dependent)      | **0.5976** |
|                            |                          |                           | - Integrated Brier Score            | **0.0339** |
|                            |                          |                           | - Negative Bernouli log-likelihood  | **0.1463** |
|                            |                          |                           |                                     |          |
|                            |                          | Semi pre-trained          | Competing-risk fine-tuning          |          |
|                            |                          |                           | - Concordance (time-dependent)      | **0.6095** |
|                            |                          |                           | - Integrated Brier Score            | **0.0338** |
|                            |                          |                           | - Negative Bernouli log-likelihood  | **0.1461** |
|                            |                          |                           |                                     |          |
| Multi-morbidity            | Any next disease         | From scratch              | Single-risk fine-tuning             |          |
|                            |                          |                           | - Concordance (time-dependent)      | *(pending)* |
|                            |                          |                           | - Integrated Brier Score            | *(pending)* |
|                            |                          |                           | - Negative Bernouli log-likelihood  | *(pending)* |
|                            |                          |                           |                                     |          |
|                            |                          | Semi pre-trained          | Single-risk fine-tuning             |          |
|                            |                          |                           | - Concordance (time-dependent)      | *(pending)* |
|                            |                          |                           | - Integrated Brier Score            | *(pending)* |
|                            |                          |                           | - Negative Bernouli log-likelihood  | *(pending)* |

*A note on scaling:*

Each of these task can only be pre-trained on the **sub-population** of patients sharing some index criteria. For example,
the Type 2 Diabetes Mellitus population includes only those patients with this diagnosis - with the input data 
being those records up to and including this diagnosis and the outcome being the either the Hypertension token,
or a set of Cardiovascular disease related tokens, for each task respectively. This is because the **forked code 
in this repository does not scale to the full training population** of 23 million patients and 7.6 billion events 
used in SurvivEHR. Consequently, these results must be compared to training SurvivEHR from scratch (referred to as
SFT in the accompanying paper) - rather than a model that has been pre-trained on the entire population. 

---

## 3. Submodule Setup

This repository depends on other projects which are included as Git submodules (e.g. `DeSurv`).  
To ensure they are correctly downloaded and kept in sync:

1. **Clone with submodules**:
   ```bash
   git clone --recurse-submodules https://github.com/cwlgadd/BEHRT-SurvivEHR.git
	 ```
2. **If you already cloned without submodules:**
   ```bash
   git submodule update --init --recursive
   ```
   
  ```
