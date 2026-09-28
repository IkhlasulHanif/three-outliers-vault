# GEMASTIK XVI (2023): Helmet Detection Proposal and Web Attack Log Classification

> Data Mining division of GEMASTIK XVI. Our proposal paper detected helmet use on Indonesian motorcyclists with Deformable DETR. In the on-site final round we classified web-server access logs into MITRE ATT&CK tactics.

This folder holds work from both stages of the competition:

1. **Proposal / re-registration paper**: *"Deteksi Penggunaan Helm Pada Pengendara Motor di Indonesia Menggunakan Deformable DETR"* (Helmet Detection for Motorcyclists in Indonesia Using Deformable DETR).
2. **On-site final**: a tabular classification task on nginx access logs, solved on the competition PC.

We also include the text-classification notebooks we prepared before the final, because we expected an NLP task like the GEMASTIK 2022 one.

## Problem

**Proposal (helmet detection).** Indonesia's electronic traffic enforcement cameras (ETLE) cannot yet catch motorcyclists who ride without a helmet. We recorded public Bali Tower CCTV streams in Jakarta and hand-labelled 844 images in Roboflow with two classes, `helm` and `nohelm` (590 train / 170 validation / 84 test). We compared DETR with Deformable DETR variants using mAP@0.5 and measured inference FPS on a Kaggle P100.

**Final round (web attack logs).** Each row is one nginx access-log entry with fields such as `status`, `request_method`, `http_referer`, `upstream_*_time`, `connection_requests` and `remote_user`. The target `rule.mitre.tactic` has six classes. It is highly imbalanced: in the 54,480 training rows there are 52,101 `non-attack`, 1,784 `Impact`, 561 `Reconnaissance`, 29 `Defense Evasion`, 4 `Initial Access` and 1 `Credential Access`. The test set has 13,599 rows, and the submission is `id, rule.mitre.tactic`. The notebooks and our own evaluation used macro F1.

## Approach

### Proposal: Deformable DETR (see `docs/`)
- We fine-tuned DETR, Deformable DETR (DDETR), DDETR with iterative bounding-box refinement (IBBR), two-stage DDETR and two-stage DDETR + IBBR, all with lr 1e-4, 50 epochs and batch size 4, on the COCO-format dataset.
- `experiments/helmet_video_demo_mmtracking.ipynb` is our attempt to run an MMTracking video demo on a recorded CCTV clip. That run failed on an `mmcv` import.

### Final round: web attack log classification (`notebooks/`)
- **EDA and rule mining** (`experiments/eda_*`, `docs/final_round_feature_notes.docx`). We ran crosstabs of each field against the tactic. They showed values that only ever appear as `non-attack`: specific `http_referer` and `upstream_addr` values, `HEAD` requests, and the users `GapuraUb2020-Mobile` and `google-oauth-client`. They also showed strong signals: 85% of POST requests are `Impact`, and 97% of GET requests are `non-attack`.
- **Feature cleaning.** Multi-valued `upstream_connect/header/response_time` strings were reduced to their numeric maximum, and we added `hour` from the timestamp. Text and ID columns were dropped (`remote_addr`, `request`, `user_agent`, `hostname`, ...). `http_referer` was grouped into `notrealdomain` / `ipaddress` / `realwebsite` / other.
- **01: Non-attack undersampling.** We applied the rule filters, then kept the 2,379 non-attack rows with the highest mean pairwise cosine similarity, plus all attack rows. The result is `train_sampled_data.csv` with 4,758 rows. The idea came from our prep notebook `preparation/diverse_sample_selection_umap_cosine.ipynb`.
- **02: Rules + MLP.** We removed rule-covered rows, one-hot encoded `status` and `request_method`, standardised the features, and trained a 256-256-128 SELU MLP with SGD. It predicted the test set with the same preprocessing.
- **03: CatBoost + post-evaluation rules (final).** GPU CatBoost with categorical `status`, `request_method` and grouped `http_referer`, trained on the undersampled data. Its test predictions were then overwritten step by step:
  - rows matched by the mined non-attack rules (referer, HEAD, safe users, `status > 404`, upstream address) were set to `non-attack`;
  - predicted `Reconnaissance` rows took the label from an earlier non-attack/Reconnaissance prediction file;
  - rows with a Reconnaissance probability above 0.995 were set back to `Reconnaissance`.

  These steps produced submissions 1 to 5.
- **Other experiments** (`experiments/`):
  - one-hot CatBoost plus a class-weighted MLP;
  - CatBoost on the full training set;
  - CatBoost on status-404 rows only;
  - a quick 100-iteration CatBoost on the sampled data.

### Preparation for an NLP final (`preparation/`)
These notebooks use the GEMASTIK 2022 news-sentiment data (`gemastik_22.csv`: title + content → `sentimen_berita`, with Negatif/Netral/Positif labels) and the HoASA hotel aspect-sentiment data. They cover:
- IndoBERT, Indonesian RoBERTa and BigBird fine-tuning;
- QLoRA on BLOOMZ-7b1-mt;
- fastText + InSet lexicon features with CatBoost;
- a multi-output Keras model for aspect-based sentiment;
- selecting a diverse training sample with TF-IDF, UMAP and cosine similarity.

## Results

**Proposal: helmet detection (test set, from the paper)**

| Model | mAP@0.5 | Inference FPS |
|---|---|---|
| DETR | 0.385 | 23.27 |
| Deformable DETR | 0.590 | 10.98 |
| DDETR + IBBR | 0.593 | 11.12 |
| **Two-stage DDETR** | **0.611** | 11.08 |
| Two-stage DDETR + IBBR | 0.579 | 11.12 |

**Final round: hold-out validation scores recorded in the notebooks** (we have no leaderboard scores or final ranking on record)

| Notebook | Validation data | Macro F1 | Accuracy |
|---|---|---|---|
| `02_rule_filtering_mlp` (MLP) | 20% of rule-filtered full train (10,554 rows) | 0.60 | 0.99 |
| `catboost_onehot_and_weighted_mlp` (CatBoost, one-hot) | 20% of full train (10,896 rows) | 0.40 | 0.99 |
| `catboost_sampled_quick_baseline` (CatBoost, 100 it.) | 20% of undersampled data | 0.9512 | 0.9905 |
| `03_catboost_rules_final_submission` (CatBoost) | 20% of undersampled data (952 rows) | 0.99 | 0.99 |

The last two rows are measured on the undersampled data, where non-attack is no longer the dominant class. They are therefore optimistic compared with the real test distribution.

**Preparation notebooks (GEMASTIK 2022 news sentiment, 20% hold-out of 5,289 articles)**

| Notebook | Metric | Score |
|---|---|---|
| IndoBERT (`ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa`) | macro F1 / accuracy | 0.55 / 0.60 |
| Indonesian RoBERTa (`w11wo/indonesian-roberta-base-sentiment-classifier`) | macro F1 / accuracy | 0.53 / 0.56 |
| BigBird (`ilos-vigil/bigbird-small-indonesian-nli`) | weighted F1 | 0.647 |
| fastText + lexicon + CatBoost | macro F1 / accuracy | 0.53 / 0.59 |

## Repository structure

```
gemastik-xvi/
├── README.md
├── notebooks/                                   # on-site final solution (web attack log classification)
│   ├── 01_non_attack_undersampling.ipynb        # rule filters + cosine-similarity undersampling -> train_sampled_data.csv
│   ├── 02_rule_filtering_mlp.ipynb              # rule filters + Keras SELU MLP, submission_mlp_rules.csv
│   └── 03_catboost_rules_final_submission.ipynb # CatBoost + post-evaluation rule overrides (submissions 1-5)
├── experiments/
│   ├── eda_feature_vs_tactic_crosstabs.ipynb    # crosstabs used to mine the non-attack rules
│   ├── eda_test_set_and_credential_access.ipynb # train/test status codes, the single Credential Access row
│   ├── catboost_onehot_and_weighted_mlp.ipynb   # first iteration: one-hot CatBoost + class-weighted MLP
│   ├── catboost_full_train_baseline.ipynb       # CatBoost on all 54k rows, first submission
│   ├── catboost_sampled_status404.ipynb         # CatBoost on undersampled data, status 404 only
│   ├── catboost_sampled_quick_baseline.ipynb    # 100-iteration CatBoost baseline (macro F1 0.9512)
│   └── helmet_video_demo_mmtracking.ipynb       # proposal: MMTracking video demo setup (Colab)
├── preparation/                                 # NLP prep before the final (GEMASTIK 2022 data, HoASA)
│   ├── news_sentiment_indobert_finetune.ipynb
│   ├── news_sentiment_indonesian_roberta_finetune.ipynb
│   ├── news_sentiment_bigbird_finetune.ipynb
│   ├── news_sentiment_bloomz_qlora.ipynb        # QLoRA LLM fine-tuning (not fully run)
│   ├── news_sentiment_fasttext_catboost.ipynb
│   ├── hotel_absa_fasttext_multitask_nn.ipynb   # aspect-based sentiment, one head per aspect
│   └── diverse_sample_selection_umap_cosine.ipynb
├── docs/
│   ├── helmet_detection_deformable_detr_paper.docx  # our proposal paper (original title above)
│   └── final_round_feature_notes.docx               # our on-site notes per log field ("Catatan")
└── data/
    ├── sample_submission.csv                    # competition sample submission (id, rule.mitre.tactic)
    └── submissions/
        └── submission_mlp_rules.csv             # test predictions written by 02_rule_filtering_mlp
```

The paper is kept as `.docx`. `textutil` is available, but it drops the paper's figures and cannot write PDF.

We left these out of the repository: our team profile video (`VideoProfil_ThreeOutliers.mp4`), the CCTV demo recording and MMTracking checkpoint used by the video demo notebook, the competition datasets (`train.csv` about 23 MB, `test.csv` about 98 MB), the GEMASTIK 2022 dataset, and intermediate prediction CSVs.

## How to run

The final-round data was only available on the competition PC and is not public. To rerun the notebooks, put the files in `data/`:

```
data/
├── train.csv                 # competition training logs (54,480 rows)
├── test.csv                  # competition test logs (13,599 rows)
├── sample_submission.csv     # included
├── train_sampled_data.csv    # produced by notebooks/01_non_attack_undersampling.ipynb
├── noattack.csv              # earlier non-attack/Reconnaissance predictions read by notebook 03 (not preserved)
└── submissions/
```

Run the notebooks from inside their own folder, since all paths are relative to `../data`. Run order: `01` → `02` → `03`.

Main dependencies: `pandas`, `numpy`, `scikit-learn`, `catboost` (GPU), `tensorflow`/`keras`, `seaborn`, `matplotlib`.

The preparation notebooks need these extra files in `data/`:
- `gemastik_22.csv` (GEMASTIK 2022 news sentiment data);
- `positive.tsv` and `negative.tsv` from the [InSet lexicon](https://github.com/fajri91/InSet);
- `hoasa_absa-airy/` from IndoNLU (HoASA).

They also need the fastText `cc.id.300.bin` vectors from [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html) in `models/`. Their Python dependencies are `transformers`, `torch`, `nlp-id`, `Sastrawi`, `gensim`, `stanza`, `umap-learn`, `peft`, `trl` and `bitsandbytes`.

## Team
Three Outliers: Eduardus Tjitrahardja, Ikhlasul Akmal Hanif, Rahmat Bryan Naufal
