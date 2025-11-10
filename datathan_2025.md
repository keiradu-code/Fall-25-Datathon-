# Datathon 2025

## Business problem

You are a junior data scientist at **FutureBright Insurance**. The underwriting department of the Automobile business line has requested your team's assistance in building a risk segmentation model using historical auto policy exposure and claims data. The goal of this project are threefold:

1. **Quantify expected losses** — and thus the risk level — for potential customers.
2. **Support the underwriting and actuarial departments** by providing insights into the nature of the business, such as:
   - Identify which segments of the portfolio are underpriced or overpriced.
   - Offer strategies to adjust rates accordingly.
3. **Assist the marketing department** in designing targeted campaigns to engage prospective customers.

## Data

Your team has received two data sets for this initiative.

- Model data: Contains exposure and claims information of 15000 historical policy terms. The model dataset is available at https://drive.google.com/file/d/1Wvf4WmgOVFLu3snFkIUYI6Y5unM9b5o1/view?usp=drive_link.
- Inference: Contains information for 15000 future potential customers. The inference dataset is available at https://drive.google.com/file/d/17j2pkUwyUdOWg3-2xKQReL2JVHtSbGSN/view?usp=drive_link.

In the modeling dataset, key claim-related fields include:

- `claimcst0` - claim cost
- `clm` - claim indicator
- `numclaims` - claim counts

In the inference data, these fields are absent due to their unavailability for future customers.

For detailed definitions of each field, please refer to the [Data Dictionary](https://drive.google.com/file/d/1tfFlSwFv6wVCa6uhFXHTDrfLNiKQtz6c/view?usp=sharing).

For instructions on downloading the data, refer to the sample code in the _Initialize_ section at the beginning of this notebook in the repo: analysis_pipeline/1_data_etl.ipynb.

## Modeling

After reviewing the data, your manager - a lead data scientist - decides to develop a quantified machine learning model to predict the claim cost per policy term ( `claimcst0`). You are asked to follow standard modeling steps on this effort starting from model design, data collection and cleaning, data exploration, model selection and model implementation.

In addition, your manager encourages initiative and experimentation, and recommends testing the following techniques:

#### 1. Variable reduction

Variabel reduction is often a pain point in modeling, particularly in the era of big data. In previous roles, your manager used a principle component based hierarchical variable reduction technique in SAS called `VARCLUS` ([support.sas.com/documentation/onlinedoc/stat/132/varclus.pdf](https://support.sas.com/documentation/onlinedoc/stat/132/varclus.pdf)) to conduct efficient variable reduction. Recently, he found a Python package implementing a similar approach - `VarClusHi` ([jingtt/varclushi: A Python package for variable clustering](https://github.com/jingtt/varclushi?tab=readme-ov-file)), which he would like you to try.

For an introduction of this method, refer to this notebook: [data_exploration.ipynb](https://github.com/mengjin67/data_science_bootcamp_lecture_1/blob/9487a900ff3be60716a8c534c96207bc399e5ea8/analysis_pipeline/data_exploration.ipynb).

You are also welcome to explore alternative methods, such as penalized regresion. Regardless of the approach, your final delivery should include a detailed description of the method used on variable reduction.

#### 2. Tree-based models & Hyper-parameter Tunning

Tree-based models(Gradient Boosting Machines and Random Forests) are widely used in the finance and insurance due to their high predictive accuracy. In this project, your manager decides to use the tree-based model as the primary modeling technique. However, there is a ton of hyper-parameters in the tree-based models.

Your manager suggest moving away from exhausive grid/random search due to its inefficiency. Instead, you are encouraged to consider an interactive hyperparameter tuning approach - tuning two or three hyperparameters at a time - which allows for both human insight and better computational efficiency.

Tutorial on this topic:

- [interactive hyper-parameter tunning blog 1](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/).
- [interactive hyper-parameter tunning blog 2](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/). Refer to [model_selection.ipynb](https://github.com/mengjin67/data_science_bootcamp_lecture_1/blob/9487a900ff3be60716a8c534c96207bc399e5ea8/analysis_pipeline/model_selection.ipynb) for more details.

Whatever method you choose, please include a detailed description of the hyperparameters and tuning strategy.

#### 3. Foundation Models for Tabular Data

Due to the popularity of large language models, transformer architectures have gain popularity. A 2025 article in _Nature_ titled "Accurate predictions on small data with a tabular foundation model" presents `TabPFN`, a transformer-based model that claims superior performance on small to medium size tabular datasets.

In this project, you may explore and compare `TabTPN`'s performance against the tree-based models built.

Resources:

- `TabPFN` paper: [https://arxiv.org/abs/2207.01848](https://arxiv.org/abs/2207.01848).
- [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

Refer to [model_selection.ipynb](https://github.com/mengjin67/data_science_bootcamp_lecture_1/blob/9487a900ff3be60716a8c534c96207bc399e5ea8/analysis_pipeline/model_selection.ipynb) for some more notes.

If you choose to use `TabPFN`, please include a detailed explanation of this method, along with its strengths and limitations, in the final delivery.

#### 4. Frequency-Severity Modeling & Exposure Handling

Estimating claim cost per policy term is complex due to two main challenges:

- The total claim cost is a product of freqnency and severity.
- Policy terms have varying exposure durations.

To address these:

- Use a composite modeling approach: build separate models for frequency and severity, then combine their predictions.
- Account for variable exposures using weighted likelihood estimation or offset techniques.

For more details, refer to [model_selection.ipynb](https://github.com/mengjin67/data_science_bootcamp_lecture_1/blob/9487a900ff3be60716a8c534c96207bc399e5ea8/analysis_pipeline/model_selection.ipynb).

If you explore any of these methodologies, be sure to include detailed documentation of your approach.

#### 5. Model Explainability

Machine learning models are often criticized as "black box" due to their lack of interpretability. One powerful method for explainability is `SHAP` (Shapley Additive Explanations), which quantifies the contribution of each feature to individual predictions as well as the overall model.

Resources:

- `SHAP` paper: [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874).
- [Explaining XGBoost model predictions with SHAP values — SAMueL - Stroke Audit Machine Learning](https://samuel-book.github.io/samuel_shap_paper_1/xgb_10_features/03_xgb_combined_shap_key_features.html).

Please use `SHAP`, or similar tools, to explain your model results - this will be particularly helpful for addressing business questions.

## Final Delivery

Your final delivery should include two parts, with the focus on part 1.

1. Internal review (Technical Audience)
   - Target: Data scientists
   - Purpose: To describe the modeling procedures and techniques used at each stage, with particular emphasis on data exploration and model selection. Include both qualitative discussion and quantitative results from your analysis.
   - Note:
     - In the model data, the building and validation samples can be filtered using the `sample` column. Please use the rows where `sample=='1|bld'` to train the model, and the rows where `sample=='2|val'` to evaluate or test the model performance. Please avoid over-using the validation samples.
     - You do not need to include every modeling technique discussed in the Modeling section above, please decide based on your interest and availability, and the fitness of the techniques to the data.
2. External review (Business Audience)
   - Target: Stakeholders (underwriters, actuaries, sales team)
   - Purpose: To address the original business questions by providing clear and actionable recommendations on prospective customers based on the inference data .

## Note

The data used in this datathon is synthetic, derived from the dataset used in the Kaggle contest: [2023 UMN Travelers Analytics Case Competition](https://www.kaggle.com/competitions/2023-umn-travelers-analytics-case-competition/datahttps://www.kaggle.com/competitions/2023-umn-travelers-analytics-case-competition/data).
