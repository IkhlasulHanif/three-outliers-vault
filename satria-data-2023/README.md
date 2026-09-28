# Satria Data (2023): Indonesian Vehicle License Plate Recognition

> Read the registration number from cropped images of Indonesian vehicle license plates (Big Data Challenge, preliminary round).

## Problem
The preliminary round ("Penyisihan") of the Satria Data 2023 Big Data Challenge (BDC) gave teams images of Indonesian vehicle license plates, and the task was to output each plate's registration number as a string (for example `B1661TKZ`). The labelled training images come with a `DataTrain.csv` file (`Vehicleregistrationplate`, `NameofFile`). The test set has 100 images (`DataTest1.png` to `DataTest100.png`). We used a labelled copy of the test set (`data/data_test_labeled.csv`) for local evaluation, scoring predictions with a per-character positional accuracy (the share of label positions where the predicted character matches), averaged over images.

## Approach
We tried three routes:

- **Character detection with Faster R-CNN** (`experiments/`): every character on the training plates is annotated in Pascal VOC XML (37 classes: background, `0`–`9`, `A`–`Z`). We then fine-tuned torchvision's `fasterrcnn_resnet50_fpn` with a new `FastRCNNPredictor` head (SGD, lr 0.005, momentum 0.9, weight decay 0.0005, 5 epochs, batch size 8, 64/16/20 train/val/test split). There are two variants: images resized to 224x224 with albumentations, and images at their original size. With the default decoding the detector returned no confident boxes on the held-out split, so we did not pursue this route further.
- **Fine-tuning keras-ocr** (`notebooks/keras_ocr_plate_recognition.ipynb`):
  - We fine-tuned the CRAFT detector on the character boxes (imgaug affine, blur and brightness augmentation; 640x640; early stopping).
  - We fine-tuned the CRNN recognizer (`crnn_kurapan`) on the whole plate images with their lowercase plate strings (GammaContrast augmentation, 80/20 split, batch size 8, early stopping with patience 10, best checkpoint by `val_loss`).
  - We evaluated the recognizer on the 100 labelled test images, before and after fine-tuning.
- **TrOCR** (final predictions): we fine-tuned `microsoft/trocr-base-printed` (`VisionEncoderDecoderModel`, transformers 4.30.1, beam search with 4 beams, `max_length` 64). One of the runs was named `tr-ocr-fulldata-gray-blur`, which suggests training on the full data with grayscale and blur preprocessing. The TrOCR training notebook is not in the archive. Only its local test predictions are kept (`data/trocr_test_predictions_local_acc095.csv`).

## Results
All scores are on the labelled 100-image test set, using the per-character positional accuracy described above.

| Model | Local char accuracy | Exact-match plates | Source |
|---|---|---|---|
| keras-ocr recognizer, pretrained | qualitatively poor (e.g. `aotose` vs `ad7034oe`) | – | notebook |
| keras-ocr recognizer, fine-tuned | 0.7145 | – | `notebooks/keras_ocr_plate_recognition.ipynb` |
| TrOCR (`trocr-base-printed`), fine-tuned | **0.9518** | 82 / 100 | `data/trocr_test_predictions_local_acc095.csv` |

The best keras-ocr recognizer checkpoint reached val_loss 6.35 at epoch 9 (`data/keras_ocr_recognizer_training_log.csv`).

## Repository structure
```
satria-data-2023/
├── README.md
├── notebooks/
│   └── keras_ocr_plate_recognition.ipynb          # fine-tunes keras-ocr detector + recognizer, evaluates on labelled test set
├── experiments/
│   ├── faster_rcnn_char_detection_resize224.ipynb # Faster R-CNN per-character detector, images resized to 224x224
│   └── faster_rcnn_char_detection_noresize.ipynb  # same detector trained on original-size images
├── data/
│   ├── data_test_labeled.csv                      # labels for the 100 test images (Name of File, label)
│   ├── trocr_test_predictions_local_acc095.csv    # TrOCR predictions vs labels, per-image accuracy
│   ├── keras_ocr_recognizer_training_log.csv      # CSVLogger output of the keras-ocr recognizer fine-tuning
│   └── semifinal_regulation_text_db.xlsx          # semifinal: Indonesian regulations split into chapter/article/paragraph rows
└── references/
    └── pmk_1_pmk05_2021.pdf                       # Peraturan Menteri Keuangan No. 1/PMK.05/2021 (semifinal source document)
```

### Semifinal material
The semifinal folder held no code. It had `text_db.xlsx`, a table of Indonesian laws and regulations (UU 1/2004, Perpu 1/2020, PP 49/2021, PP 31/2022, several Inpres, KMK/Kepmenko, and PMK 1/PMK.05/2021) split into one row per paragraph. The columns are `nama_file`, `hierarki`, `no_bab`, `judul_bab`, `no_bagian`, `judul_bagian`, `no_pasal`, `no_ayat` and `isi_ayat`, with one sheet per regulation plus a `combined` sheet. `references/pmk_1_pmk05_2021.pdf` is the original text of *Peraturan Menteri Keuangan Nomor 1/PMK.05/2021 tentang Tarif Layanan Badan Layanan Umum Pusat Investasi Pemerintah pada Kementerian Keuangan*. It is a public Indonesian government regulation published by the Ministry of Finance (Kementerian Keuangan RI), and is included only as source material.

## How to run
The dataset belongs to the Satria Data 2023 organisers and is not included here. Put it under `data/` next to the small files, using this layout (the notebooks read it from `../data`):

```
data/
├── DataTrain.csv            # organiser labels (sep=";"): Vehicleregistrationplate, NameofFile
├── data_test_labeled.csv    # included
├── train images/            # DataTrain<N>.png
├── train annotations/       # DataTrain<N>.xml, Pascal VOC boxes, one <object> per character (name = 0-9 / A-Z)
├── train images - Copy/     # copy of the training images, used only for the before/after preview in the keras-ocr notebook
└── test images/             # DataTest1.png ... DataTest100.png
```

Our archived copy of the data contains:
- `train images sr padding/`: 435 PNGs, super-resolved and padded training images.
- `test images sr cropped/`: 100 PNGs, super-resolved and cropped test images.
- `train annotations/`: 521 character-level VOC XMLs.
- `train annotations plate/`: 336 VOC XMLs with a single `plate` box per image.

Rename or symlink the image folders to the names above before running the notebooks. Trained weights (`models/*.pth`, `models/keras-ocr/*.h5`, TrOCR checkpoints) are not included. The notebooks write them to `../models`.

Main dependencies:
- Faster R-CNN: `torch`, `torchvision`, `albumentations`, `torch_snippets`.
- keras-ocr: `keras-ocr`, `tensorflow`, `imgaug`, `opencv-python`.
- Both: `pandas`, `scikit-learn`, `matplotlib`.

The keras-ocr notebook was run on Windows, and its file-sorting key splits paths on `\\`.

## Team
Three Outliers: Eduardus Tjitrahardja, Ikhlasul Akmal Hanif, Rahmat Bryan Naufal
