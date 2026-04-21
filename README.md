# DeepTruth — LSTM Misinformation Detector

## Project Structure

```
DeepTruth_Project/
├── data/
│   └── README_dataset.txt      ← dataset download instructions
├── models/
│   ├── fake_news_model.h5      ← saved after training
│   └── tokenizer.pkl           ← saved after training
├── src/
│   ├── preprocess.py           ← cleaning, tokenising, padding, splitting
│   ├── model.py                ← LSTM architecture
│   ├── train.py                ← training loop + plots
│   ├── evaluate.py             ← metrics + confusion matrix
│   └── interpret.py            ← LIME word-level explanations
├── app/
│   └── app.py                  ← Streamlit web app
├── outputs/
│   ├── plots/                  ← accuracy_plot.png, loss_plot.png
│   └── confusion_matrix.png
├── requirements.txt
├── run.sh
└── README.md
```

---

## 1. Setup

```bash
# Clone / unzip project, then:
cd DeepTruth_Project
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

---

## 2. Download Dataset

Follow `data/README_dataset.txt`.  Quick version:

```bash
pip install kaggle
# place kaggle.json at ~/.kaggle/kaggle.json first
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
unzip fake-and-real-news-dataset.zip -d data/
```

Expected files: `data/Fake.csv`, `data/True.csv`

---

## 3. Train

```bash
python src/train.py
```

Outputs saved to:
- `models/fake_news_model.h5`
- `models/tokenizer.pkl`
- `outputs/plots/accuracy_plot.png`
- `outputs/plots/loss_plot.png`

Hyperparameters (edit `src/train.py` / `src/model.py`):
| Parameter | Default |
|-----------|---------|
| MAX_WORDS | 10 000  |
| MAX_LEN   | 200     |
| EPOCHS    | 10      |
| BATCH_SIZE| 64      |
| LSTM units| 64      |
| Dropout   | 0.5     |

---

## 4. Evaluate

```bash
python src/evaluate.py
```

Prints accuracy, precision, recall, F1 and saves `outputs/confusion_matrix.png`.

---

## 5. Interpret (LIME)

```bash
python src/interpret.py
```

Prints top-10 influential words for two demo articles and saves
`outputs/lime_explanation.html`.

To explain custom text, import and call from Python:

```python
from src.interpret import explain
explain("Your article text here", num_features=10)
```

---

## 6. Run Streamlit App

```bash
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

---

## 7. One-Shot Runner

```bash
bash run.sh                    # train + evaluate + launch app
bash run.sh --skip-train       # skip training (model must already exist)
bash run.sh --no-app           # train + evaluate only
```

---

## 8. Google Colab

```python
# In a Colab cell:
!pip install -r requirements.txt
!python src/train.py
!python src/evaluate.py
# Streamlit needs a tunnel in Colab — use localtunnel or pyngrok:
!pip install pyngrok
from pyngrok import ngrok
import subprocess, time
proc = subprocess.Popen(["streamlit", "run", "app/app.py", "--server.port=8501"])
time.sleep(3)
print(ngrok.connect(8501))
```

---

## Labels

| Label | Meaning |
|-------|---------|
| 0     | Fake    |
| 1     | Real    |
