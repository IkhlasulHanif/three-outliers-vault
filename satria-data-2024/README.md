# Satria Data 2024, Big Data Challenge (2024): Indonesian Election Tweet Classification and Lexical-Mutation Campaign Analysis

> Classify Indonesian tweets about the 2024 presidential election into eight national-resilience topics, then analyse amplified campaigns, stance and civility on X. **Result:** 2nd place (Juara 2), Big Data Challenge (BDC), Satria Data 2024 (Statistika Ria dan Festival Sains Data), Universitas Telkom, 29 July to 2 August 2024.

## Problem

**Preliminary round.** Each tweet (`IDText`, `Text`) must be labelled with one of eight *Astagatra* classes: Politik, Sosial Budaya, Ekonomi, Pertahanan dan Keamanan, Ideologi, Sumber Daya Alam, Demografi, Geografi. The training set is heavily imbalanced: after cleaning it holds 2,968 Politik tweets but only 60 Demografi and 17 Geografi. The test set has 1,000 tweets, and the submission format is `IDText;Kelas`. We tracked **balanced accuracy** on a stratified 80/20 hold-out split.

**Semifinal and final.** The task was an open-ended analysis of about 9.8 million hashed tweets about the 2024 presidential election (January 2024). We wrote the paper *"Analisis Kampanye Politik Menggunakan Lexical Mutations pada Pemilu Presiden Indonesia 2024"* and presented *"Analisis Kampanye Pemilu 2024 menggunakan Jaringan Lexical Mutations, Stance Publik dan Kecenderungan Civility di Media Sosial"* in the final.

## Approach

### Preliminary round: tweet topic classification (`notebooks/01`–`05`)
- **Cleaning:** remove mentions, `RT`, `[RE …]` tags, URLs, emojis and non-alphanumeric characters, fix `&amp;` and leetspeak (`m3nang` → `menang`), expand slang and abbreviations with a word dictionary, lowercase, and drop duplicated texts. Training rows that we flagged as mislabelled (`is_false` in `train_fix.csv`) are removed.
- **Keyword features:** 1,764 class keywords (`data/keywords_prob.xlsx`, `data/daftar_kata.csv`) become binary "contains keyword" columns.
- **CatBoost:** a multiclass `CatBoostClassifier` (GPU, `TotalF1` eval metric, class weights `n / (k · n_class)`) on the raw text as a CatBoost text feature plus the keyword columns. We ran it on both raw and lemmatised text (`nlp_id` lemmatizer).
- **E5 similarity matching:** `intfloat/multilingual-e5-large` embeddings of train and test tweets. For each test tweet we find the most similar training tweet by cosine similarity. Near-duplicates (for example `Politik` with similarity ≥ 0.98) take the training label, and a list of duplicated texts (`data/test_leakage.csv`) overrides predictions.
- **IndoBERTweet fine-tuning:** `indolem/indobertweet-base-uncased` and `fathan/ijelid-ft-indojave-indobertweet`, trained with a class-weighted cross-entropy `Trainer` and early stopping.
- **Rule-based ensemble:** IndoBERTweet predictions form the base. Geografi comes from the CatBoost baseline. Per-class similarity thresholds (Ekonomi 0.95, Geografi 0.909, Ideologi 0.938, Pertahanan dan Keamanan 0.952, Sosial Budaya 0.942, Politik 0.98) replace labels, and Ekonomi, Sosial Budaya or Ideologi is kept when CatBoost, IJELID and the baseline all agree. Two keyword rules finish the ensemble: "impor sapi" maps to Sumber Daya Alam and "bloomberg" maps to Politik.
- **Explored but not used in the final pipeline** (`experiments/`): one-vs-all CatBoost per class, translating tweets to English with Google Translate and fine-tuning English models (BERT, RoBERTa, BART and others, with Optuna and `nlpaug` contextual augmentation), TweetNLP and sentence-transformer embeddings as CatBoost features, CatBoost on fine-tuned IndoBERTweet embeddings, and zero-shot labelling with GPT-4o through LangChain.

### Semifinal and final: election campaign analysis (`notebooks/semifinal/`)
- **Data reduction:** keep non-retweets with more than 10 retweets and drop duplicates. This leaves 17,182 tweets out of 9,817,355 (January 2024).
- **Lexical-mutation network** (after Phadke & Mitra, 2024): embed tweets with `intfloat/multilingual-e5-base`, compute pairwise cosine similarity, connect pairs above **0.925**, detect communities with **Louvain**, and visualise them with Gephi and pyvis. Each community is one campaign.
- **Stance and civility:** we annotated about 1,500 tweets by hand for aspect-based stance (Pro / Kontra / Netral per candidate pair, input `sentence[SEP]aspect[SEP]`) and civility (Sopan / Vulgar / SARA). We fine-tuned transformer classifiers with weighted cross-entropy and used the best one, IndoBERTweet, to label the remaining tweets.
- **Analysis:** stance and civility over time around key events (debates, Kadin and KPK dialogues), per candidate and per campaign cluster, plus hashtag word clouds per candidate.

## Results

### Preliminary round: hold-out balanced accuracy (from notebook outputs)

| Model | Notebook | Balanced accuracy |
|---|---|---|
| CatBoost text + keyword features, lemmatised text | `notebooks/03_catboost_lemmatized_keywords.ipynb` | 0.5367 |
| **CatBoost text + keyword features** | `notebooks/02_catboost_keyword_baseline.ipynb` | **0.6057** |
| IndoBERTweet (`indolem/indobertweet-base-uncased`), weighted loss | `notebooks/04_indobertweet_finetuning.ipynb` | 0.5374 |
| IJELID IndoBERTweet (`fathan/ijelid-ft-indojave-indobertweet`), weighted loss | `notebooks/04_indobertweet_finetuning.ipynb` | 0.5530 |
| IndoBERTweet, weighted loss (15% hold-out, different split) | `experiments/indobertweet_weighted_loss_catboost_embeddings.ipynb` | 0.6308 |
| CatBoost on text + keywords (no lemmatisation, earlier cleaning) | `experiments/catboost_keyword_features.ipynb` | 0.5126 |
| CatBoost on lemmatised text + keywords (main model of the one-vs-all notebook) | `experiments/catboost_one_vs_all_lemmatized.ipynb` | 0.5739 |
| Sentence-transformer embeddings + CatBoost | `experiments/sentence_transformer_embeddings_catboost.ipynb` | 0.4816 |
| Best English-translated transformer (`FacebookAI/roberta-base` + augmentation) | `data/experiment_scores.xlsx` | 0.4712 |
| GPT-4o zero-shot (138 tweets, plain accuracy) | `experiments/llm_gpt4o_zero_shot_labeling.ipynb` | 0.57 accuracy |

The hold-out splits differ between notebooks, so treat these numbers as rough comparisons. The leaderboard score of the final submission is not recorded in the vault.

### Final round: stance and civility classifiers (F1, from the final presentation)

| Model | F1 stance | F1 civility |
|---|---|---|
| IndoBERT-1 | 0.363 | 0.724 |
| IndoBERT-2 | 0.722 | 0.800 |
| RoBERTa-1 | 0.678 | 0.734 |
| RoBERTa-2 | 0.719 | 0.718 |
| **IndoBERTweet** | **0.768** | **0.834** |

Key findings from the deck:
- The network holds **8,893 campaigns** built from 17,182 tweets, and on average 96% of a campaign cluster consists of unique lexical mutants.
- Prabowo–Gibran drew the largest share of *Kontra* content, and *Kontra* content spiked after the third debate.
- More than 25% of *Kontra* tweets contain vulgar or SARA content.

## Repository structure

```
satria-data-2024/
├── README.md
├── notebooks/                                   # final pipelines
│   ├── 01_e5_text_embedding.ipynb               # multilingual-e5-large embeddings of train/test tweets (pickled)
│   ├── 02_catboost_keyword_baseline.ipynb       # CatBoost + keyword features + E5 similarity + leakage fix -> submission_baseline.csv, similarity_data.csv
│   ├── 03_catboost_lemmatized_keywords.ipynb    # CatBoost on lemmatised vs raw text -> pred_catboost.csv
│   ├── 04_indobertweet_finetuning.ipynb         # IndoBERTweet + IJELID fine-tuning -> submission_indobert_edu.csv, submission_ijelid.csv
│   ├── 05_ensemble_submission.ipynb             # rule-based ensemble of the files above -> submission_final.csv
│   └── semifinal/
│       ├── 01_eda_full_tweet_dataset.ipynb      # EDA of the 9.8M-tweet dataset (tweets per day/hour, duplicates, domains)
│       ├── 02_sample_candidate_tagging.ipynb    # tag a sample with the candidate(s) it mentions
│       ├── 03_hashtag_counts_per_candidate.ipynb# hashtag counts per candidate for word clouds
│       ├── 04_lexical_mutant_e5_embeddings.ipynb# clean non-RT tweets and embed them with multilingual-e5-base
│       ├── 05_lexical_mutant_similarity_graph.ipynb # pairwise cosine similarity -> lexical-mutation graph
│       ├── 06_lexical_mutant_network_pyvis.ipynb# interactive community graph with pyvis
│       └── 07_stance_civility_visualization.ipynb # stance/civility timelines, pies and word clouds per cluster
├── experiments/                                 # iterations explored during the preliminary round
│   ├── ensemble_submission_draft.ipynb          # first version of the ensemble (combines submission files)
│   ├── catboost_keyword_features.ipynb          # CatBoost: text vs keywords vs both
│   ├── catboost_translated_text.ipynb           # CatBoost on English-translated text
│   ├── catboost_one_vs_all_lemmatized.ipynb     # one-vs-all CatBoost per class (lemmatised text)
│   ├── catboost_one_vs_all_kfold.ipynb          # one-vs-all CatBoost per class with stratified k-fold
│   ├── text_cleaning_translation_catboost.ipynb # cleaning, hashtag normalisation, translation, CatBoost
│   ├── indobertweet_weighted_loss_catboost_embeddings.ipynb # IndoBERTweet (weighted loss; focal loss also defined) + CatBoost on its embeddings
│   ├── indobertweet_embeddings_catboost.ipynb   # IndoBERTweet on cleaned text + CatBoost on its embeddings
│   ├── ijelid_indobertweet_embeddings_catboost.ipynb # IJELID IndoBERTweet + CatBoost on embeddings + keywords
│   ├── sentence_transformer_embeddings_catboost.ipynb # sentence-transformer embeddings as CatBoost features
│   ├── tweetnlp_embedding_catboost.ipynb        # TweetNLP tutorial adapted to our data (tweet embeddings + CatBoost)
│   ├── transformer_finetuning_translated_english.ipynb # HF classifiers fine-tuned on translated tweets
│   ├── english_transformer_tuning_and_augmentation.ipynb # BERT/RoBERTa schedulers, Optuna, nlpaug augmentation
│   ├── contextual_word_augmentation.py          # nlpaug BERT contextual-substitution augmentation snippet
│   └── llm_gpt4o_zero_shot_labeling.ipynb       # GPT-4o zero-shot labelling via LangChain
├── data/
│   ├── sample_submission.csv                    # submission format (IDText;Kelas)
│   ├── keywords_prob.xlsx                       # 1,764 keywords with per-class indicator
│   ├── daftar_kata.csv                          # the same keyword list, used by notebook 02
│   ├── test_leakage.csv                         # test tweets that duplicate a labelled training tweet (label_dup)
│   ├── experiment_scores.xlsx                   # log of transformer experiments and their hold-out scores
│   ├── submissions/
│   │   ├── final78_indobert_leakage.csv         # IndoBERTweet + leakage fix (read as `final78.csv` in the draft ensemble)
│   │   └── catboost_keyword_similarity_leakage.csv # CatBoost + similarity + leakage (written as `submission_gaming.csv`)
│   └── semifinal/                               # hashtag counts per candidate (output of semifinal/03)
├── docs/
│   ├── semifinal_paper_lexical_mutations_pemilu_2024.pdf # semifinal paper
│   └── final_presentation_satria_data_2024.pptx          # final-round presentation
├── assets/
│   ├── certificates/                            # 2nd place certificates for each team member
│   └── photos/                                  # award ceremony and team photos
└── references/
    └── topic_shifts_politicization_social_media_icwsm2024.pdf # Locatelli et al., ICWSM 2024 (third-party)
```

Original document titles: the semifinal paper is *"Analisis Kampanye Politik Menggunakan Lexical Mutations pada Pemilu Presiden Indonesia 2024"* (submission ID SD2024040000274). The final deck is *"Three Outliers - Final Satria Data 2024"*.

## How to run

The competition data comes from the Satria Data 2024 BDC organisers (Puspresnas / BPTI) and is not redistributed here. Put the files in `data/` so that the relative paths in the notebooks resolve (every notebook defines `DATA_DIR` in its first code cell):

```
data/
├── bdc_train.csv / train.csv          # training tweets (sep=";"; columns text, label)
├── bdc_test.csv / test.csv            # test tweets (sep=";"; columns IDText, Text)
├── train_fix.csv                      # training set with our is_false mislabel flags (IndoBERT notebook reads "train_fix - train.csv")
├── train_cleaned_v3.csv               # cleaned training text used by the IndoBERTweet notebook
├── keywords_prob.xlsx, daftar_kata.csv, test_leakage.csv   # included
└── with_embed_full.pkl, with_embed_test_full.pkl           # produced by notebooks/01
```

Run `notebooks/01` → `02` → `03` → `04` → `05`. Notebook 04 was run on Kaggle (T4 GPU); the others were run on Google Colab (GPU needed for CatBoost `task_type="GPU"` and E5).
The semifinal notebooks expect the organisers' `encrypted_merge.csv` and the derived files (`df_with_pred.csv`, `node_details_925_gephi_count.csv`, …) in `data/semifinal/`. These files are several hundred MB and are not included.

Main dependencies: `pandas`, `numpy`, `scikit-learn`, `catboost`, `transformers`, `datasets`, `torch`, `tensorflow` (GPU check only), `sentence-transformers`, `nlp_id`, `deep_translator`, `optuna`, `nlpaug`, `tweetnlp`, `langchain` / `langchain_openai` (set `OPENAI_API_KEY`), `networkx`, `pyvis`, `wordcloud`, `matplotlib`, `seaborn`.

## Credits

- `experiments/tweetnlp_embedding_catboost.ipynb` starts from the official [TweetNLP introduction notebook](https://github.com/cardiffnlp/tweetnlp) by Cardiff NLP. Our additions are the embedding and CatBoost cells at the end.
- `references/topic_shifts_politicization_social_media_icwsm2024.pdf`: Locatelli, M. S., Calais, P., Miranda, M. P., Junho, J. P., Muniz, T. L., Meira Jr., W., & Almeida, V. *Topic Shifts as a Proxy for Assessing Politicization in Social Media*. ICWSM 2024.
- The lexical-mutation idea follows Phadke & Mitra (2024), as cited in our paper.

## Team

[Three Outliers](../README.md#team): [Eduardus Tjitrahardja](https://www.linkedin.com/in/edutjie/), [Ikhlasul Akmal Hanif](https://www.linkedin.com/in/ikhlasul-akmal-h/), [Rahmat Bryan Naufal](https://www.linkedin.com/in/rahmat-bryan-naufal/)
