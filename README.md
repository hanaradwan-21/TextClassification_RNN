# 📰 BBC News Text Classification with TensorFlow

A multi-class text classification project that categorizes BBC news articles into their topics (business, entertainment, politics, sport, tech) using a neural network built with **TensorFlow / Keras**.

The notebook walks through the full NLP pipeline: downloading the data, cleaning the text, tokenizing and padding, building the model, training, and visualizing the results.

---

##  Project Structure

```
.
├── TextClassification_RNN.ipynb   # Main notebook (Google Colab)
└── README.md
```

> **Note:** Despite the notebook's file name (`RNN`), the model used is an Embedding + Global Average Pooling network (no recurrent layers).

---

##  Dataset

- **Source:** [BBC News dataset](https://storage.googleapis.com/learning-datasets/bbc-text.csv)
- **Size:** 2,225 articles
- **Columns:** `category` (label) and `text` (article body)
- **Classes:** 5 categories — `business`, `entertainment`, `politics`, `sport`, `tech`

The dataset is downloaded automatically inside the notebook using `wget`.

---

##  Pipeline

1. **Download** the dataset to `/tmp/bbc-text.csv`.
2. **Define hyperparameters** and a list of 153 English stopwords.
3. **Parse and clean** the CSV: separate labels from text and remove stopwords.
4. **Split** the data: 80% training (1,780 samples) / 20% validation (445 samples).
5. **Tokenize and pad** the text using Keras `Tokenizer` and `pad_sequences`.
6. **Encode labels** with a separate `Tokenizer`.
7. **Build and compile** the model.
8. **Train** for 30 epochs.
9. **Visualize** accuracy and loss curves.

---

##  Hyperparameters

| Parameter         | Value    | Description                                      |
|-------------------|----------|--------------------------------------------------|
| `vocab_size`      | 1000     | Max number of words kept in the vocabulary       |
| `embedding_dim`   | 16       | Size of each word embedding vector               |
| `max_length`      | 120      | Max sequence length (padding / truncating)       |
| `trunc_type`      | `post`   | Truncate from the end                            |
| `padding_type`    | `post`   | Pad at the end                                   |
| `oov_tok`         | `<OOV>`  | Token for out-of-vocabulary words                |
| `training_portion`| 0.8      | Train / validation split                         |
| `num_epochs`      | 30       | Number of training epochs                        |

---

##  Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=(max_length,)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

| Layer                    | Output Shape  | Params |
|--------------------------|---------------|--------|
| Embedding                | (None, 120, 16) | 16,000 |
| GlobalAveragePooling1D   | (None, 16)    | 0      |
| Dense (ReLU)             | (None, 24)    | 408    |
| Dense (Softmax)          | (None, 6)     | 150    |
| **Total**                |               | **16,558** |

> The output layer has 6 units because Keras' `Tokenizer` labels start from index 1 (index 0 is unused), so 5 classes map to labels 1–5.

---

##  Results

After 30 epochs:

| Metric              | Training | Validation |
|---------------------|----------|------------|
| Accuracy            | ~99.5%   | ~94.8%     |
| Loss                | ~0.059   | ~0.177     |

The accuracy and loss curves are plotted at the end of the notebook with Matplotlib.

---

##  How to Run

### Option 1: Google Colab (recommended)
1. Open `TextClassification_RNN.ipynb` in [Google Colab](https://colab.research.google.com/).
2. (Optional) Enable GPU: `Runtime > Change runtime type > GPU`.
3. Run all cells: `Runtime > Run all`.

### Option 2: Locally
```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install tensorflow numpy matplotlib jupyter

# Download the dataset (the notebook uses /tmp/bbc-text.csv)
wget https://storage.googleapis.com/learning-datasets/bbc-text.csv -O /tmp/bbc-text.csv

# Launch the notebook
jupyter notebook TextClassification_RNN.ipynb
```

---

##  Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

---

##  Possible Improvements

- Add a recurrent layer (LSTM / GRU / Bidirectional) to make use of word order, as the file name suggests.
- Use a larger vocabulary and embedding size.
- Add dropout or early stopping to reduce overfitting (training accuracy exceeds validation accuracy).
- Use pre-trained embeddings such as GloVe.
- Shuffle the data before splitting.
- Replace the deprecated `input_shape` argument with an explicit `tf.keras.Input(...)` layer.

---

##  License

This project is for educational purposes. Feel free to use and modify it.
