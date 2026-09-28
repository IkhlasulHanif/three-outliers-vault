# Intelligo ID Data Competition (2023): High-Accuracy, Low-Cost Model for Indonesian Logistics Sentiment Analysis

> Predicting late e-commerce deliveries in the preliminary round, then building a cheap but accurate sentiment classifier for Indonesian courier-app reviews (JNE and J&T) in the final. **Result:** 2nd Place Winner in the Data Competition at Intelligo ID 2023 (final score 88.125)

This folder holds work from both stages of the competition:

1. **Preliminary round**: tabular classification of e-commerce orders as late or on time.
2. **Semifinal / final round**: our own research project and paper, *"High-Accuracy, Low-Cost Model for Indonesian Logistics Sentiment Analysis"*, presented in the top-10 final ("Final 10 Besar").

## Problem

**Preliminary round (late delivery).** The organisers gave a Brazilian e-commerce dataset split into orders, customers, order items, payments and products tables (the fields match the public Olist dataset). After joining the tables, the target `is_late` is 1 when `order_delivered_timestamp` is later than `order_estimated_delivery_date`. Training has 87,427 delivered orders (80,689 on time, 6,738 late), and the test set has 38,279 orders to submit as `order_id, is_late`. Our notebooks validated with accuracy.

**Final round (logistics sentiment).** In the final we framed our own problem. Logistics companies need sentiment analysis to decide where to improve service, but fine-tuning language models such as BERT is expensive. We scraped Indonesian Google Play reviews of the JNE (61,529 reviews) and J&T (69,936 reviews) apps, labelled them positive/negative with pseudo-labelling, and compared a fine-tuned BERT with CatBoost models by weighted F1 and training time.

## Approach

### Preliminary round (`preliminary/`)
- **01, baseline.** We concatenated the five tables, dropped undelivered orders, derived `is_late`, and ran EDA on sellers, products, states, payment types and `payment_sequential`. We then trained a GPU CatBoost classifier with categorical city, state, payment type, product category, product ID and seller ID. Missing `order_approved_at` values in the test set were filled with the purchase time plus the mean approval delay.
- **02, date/time features.** We added month, day, day of week, hour and ISO week for purchase and approval, a season, product volume and weight/volume ratio, plus `deliver_diff` (delivery minus purchase, in days) and `estimated_diff`. The test set here is `test_merged_df.csv`, a copy of the test set with delivery timestamps taken from the public Olist data. The final model uses square-root balanced class weights.
- **03, Olist review features.** We joined the public Olist orders and reviews tables on purchase and approval timestamps, adding `review_score` and the review title and message. CatBoost used these as text features (10,000 iterations, depth 8, lr 0.01, early stopping 1,000).
- **Experiment**, `olist_public_data_test_label_lookup`. This notebook merges the test orders with the public Olist orders on purchase and approval timestamps, then computes `is_late` directly from the public delivery dates.

### Final round (`final/`)
- **01, scraping.** `google-play-scraper` `reviews_all` fetched every review with `lang='id'`, `country='id'` and newest-first order.
- **02, pseudo-labelling.** We kept only 1- and 5-star reviews (110,177). The seed set was 500 five-star reviews labelled positive and 500 one-star reviews labelled negative. We then fine-tuned `w11wo/indonesian-roberta-base-sentiment-classifier` in Keras (global max pooling, then Dense 128 → Dropout 0.1 → Dense 32 → softmax), using Adam (lr 5e-5, eps 1e-8, decay 0.01, clipnorm 1.0) and a batch size of 8. Predictions with more than 0.85 confidence joined the training set. After three iterations, the last model labelled the remaining 1,617 reviews. The final dataset has 109,395 reviews: 69,833 negative and 39,562 positive.
- **03, analysis.** We counted InSet lexicon positive/negative words per review, built Scattertext explorers on raw and lemmatised text, and compared average TF-IDF weights of words between positive and negative reviews.
- **04, CatBoost (final model).** We preprocessed text with `nlp_id` (lemmatisation, tokenisation, stopword removal) and added InSet positive/negative word counts. CatBoost used its built-in text features on `review_description` (GPU, 10,000 iterations, depth 8, lr 0.03, F1 eval metric, early stopping 200).
- **Experiments.**
  - `fasttext_inset_feature_extraction`: InSet counts (all words and adjectives only) plus mean fastText `cc.id.300` embeddings, computed on the first 1,000-review seed set. The paper's CatBoost + FastText variant uses this kind of feature, but its training run is not in these files.
  - `indobert_finetune_sentiment`: fine-tuning `indolem/indobert-base-uncased` on the full pseudo-labelled set. This run was stopped during epoch 2.

## Results

**Final round**, weighted F1 on the held-out split (from the paper and slides):

| Model | Weighted F1 | Training time |
|---|---|---|
| BERT (fine-tuned) | 0.90 | 7,380 s |
| CatBoost + FastText embeddings | 0.87 | 183 s |
| **CatBoost + built-in text embedding** | **0.91** | **46 s** |

- Notebook `04` reports weighted F1 0.91 and accuracy 0.91 on 21,879 validation reviews.
- In `02`, the RoBERTa labeller reached weighted F1 0.95 in both iteration 2 and iteration 3.
- Final announcement: Three Outliers took 2nd place with **88.125**. ReLu scored 88.25 (1st) and Modell Holics 87.5 (3rd); see `assets/photos/winners_announcement.jpg`.

**Preliminary round**, validation accuracy (20% stratified split):

| Notebook | Validation accuracy |
|---|---|
| 01 baseline CatBoost | 0.9357 |
| 02 + date/time features (incl. `deliver_diff`) | 0.9979 |
| 03 + Olist review score/text | 0.9527 |

No preliminary leaderboard score was found in the files.

## Repository structure

```
intelligo-2023/
├── README.md
├── preliminary/
│   ├── notebooks/
│   │   ├── 01_baseline_catboost.ipynb                   # table merge, is_late label, EDA, CatBoost baseline
│   │   ├── 02_catboost_datetime_features.ipynb          # date/time, season, product-size and delivery-gap features
│   │   └── 03_catboost_olist_review_text_features.ipynb # + public Olist review score/text as CatBoost text features
│   └── experiments/
│       └── olist_public_data_test_label_lookup.ipynb    # test labels from matching public Olist orders
├── final/
│   ├── notebooks/
│   │   ├── 01_google_play_review_scraper.ipynb          # scrape JNE / J&T Google Play reviews
│   │   ├── 02_pseudo_labeling_roberta.ipynb             # 3-iteration RoBERTa pseudo-labelling -> train_iter4.csv
│   │   ├── 03_sentiment_eda_scattertext.ipynb           # lexicon counts, Scattertext, TF-IDF word weights
│   │   └── 04_catboost_text_sentiment_classifier.ipynb  # final CatBoost text model (weighted F1 0.91)
│   └── experiments/
│       ├── fasttext_inset_feature_extraction.ipynb      # nlp_id preprocessing, InSet counts, fastText embeddings
│       └── indobert_finetune_sentiment.ipynb            # IndoBERT fine-tuning (interrupted run)
├── docs/
│   ├── semifinal_paper_logistics_sentiment_analysis.docx # semifinal paper (Indonesian)
│   └── final_presentation.pptx                          # final presentation slides (Indonesian)
├── data/
│   └── preliminary/
│       ├── sample_submission.csv                        # organiser's submission format
│       └── submission_baseline.csv                      # our baseline submission (from 01)
├── assets/
│   ├── certificates/                                    # 2nd place certificates for each member
│   └── photos/                                          # team photos, award ceremony, winners announcement
└── references/
    └── intelligo_proposal_guidelines.docx               # organiser's proposal/paper guidelines
```

Original document titles:
- `docs/semifinal_paper_logistics_sentiment_analysis.docx`: "High-Accuracy, Low-Cost Model for Indonesian Logistics Sentiment Analysis" (Three Outliers).
- `docs/final_presentation.pptx`: "Three Outliers: Final 10 Besar - Lomba Data Science Intelligo ID".
- `references/intelligo_proposal_guidelines.docx`: "Proposal Lomba Data Science Intelligo ID", written by the organiser (Intelligo ID).

## How to run

**Preliminary.** The competition data came from the organiser and is not included here. The notebooks expect this layout next to them:
- `dataset/train/df_{Orders,Customers,OrderItems,Payments,Products}.csv` and `dataset/test/...`
- the public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) files (`dataset/olist_orders_dataset.csv`, `dataset/olist_order_reviews_dataset.csv`; `other_data/olist_*.csv` for the experiment)
- `dataset/test_merged_df.csv` for notebook 02, which was built outside these notebooks

Notebook 01 writes `dataset/train_df.csv`, which notebook 02 reads.

**Final.**
1. Run `01` for each app to create `ReviewJNEIndonesia.csv` and `ReviewJ&TIndonesia.csv`, and place them in `scrape_dataset/raw/`.
2. `02` writes `scrape_dataset/train_iter{1..4}.csv`. `train_iter4.csv` (109,395 labelled reviews, about 25 MB, not included) is the input to `04` and to the IndoBERT experiment.
3. Download the InSet lexicon (`positive.tsv`, `negative.tsv`) from https://github.com/fajri91/InSet. Notebook `04` does this with `wget`.
4. For the fastText experiment, download `cc.id.300.bin` from fastText and set `FASTTEXT_MODEL_PATH`.

**Main dependencies:** pandas, numpy, scikit-learn, catboost (GPU), lightgbm, xgboost, matplotlib, seaborn, google-play-scraper, nlp-id, gensim, scattertext, tensorflow/keras, transformers, torch.

## Team

[Three Outliers](../README.md#team): [Eduardus Tjitrahardja](https://www.linkedin.com/in/edutjie/), [Ikhlasul Akmal Hanif](https://www.linkedin.com/in/ikhlasul-akmal-h/), [Rahmat Bryan Naufal](https://www.linkedin.com/in/rahmat-bryan-naufal/)
