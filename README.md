# N24News: 24-Class News Article Classification

A comparative study of traditional Machine Learning algorithms and a fine-tuned BERT transformer for multi-class news article classification using the N24News dataset.

## Project Overview

This project aims to automatically classify news articles into 24 predefined categories based on their headlines and article content.

Four classification models are implemented and evaluated:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (Linear SVM)
4. Fine-tuned BERT (`bert-base-uncased`)

The primary objective is to investigate whether a transformer-based model provides better classification performance than traditional TF-IDF-based Machine Learning approaches.

## Models and Techniques

### Traditional Machine Learning

The traditional classifiers use TF-IDF feature extraction.

**Text preprocessing:**
- Convert text to lowercase.
- Remove non-alphabetic characters.
- Normalize whitespace.
- Combine headlines and article bodies.

**TF-IDF configuration:**

| Parameter | Value |
|---|---|
| Maximum features | 20,000 |
| N-gram range | (1, 2) |
| Stop words | English |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.95 |

**Classifier configurations:**

| Model | Configuration |
|---|---|
| Multinomial Naive Bayes | Default parameters |
| Logistic Regression | `C=1.0`, `max_iter=1000` |
| Linear SVM | `C=1.0`, `max_iter=2000` |

### BERT Fine-tuning

The transformer model uses `bert-base-uncased`, fine-tuned for 24-class sequence classification.

| Parameter | Value |
|---|---|
| Pretrained model | bert-base-uncased |
| Number of labels | 24 |
| Maximum sequence length | 384 |
| Epochs | 3 |
| Learning rate | 1e-5 |
| Training batch size | 8 |
| Evaluation batch size | 16 |
| Gradient accumulation | 2 |
| Weight decay | 0.01 |
| Warmup steps | 918 |
| Mixed precision | FP16 |
| Random seed | 42 |

The best checkpoint is selected based on validation weighted F1-score.

## Dataset

The dataset contains 24 news categories.

Some of the categories include:

- Art & Design
- Automobiles
- Books
- Economy
- Education
- Fashion & Style
- Food
- Global Business
- Health
- Movies
- Music
- Science
- Sports
- Technology
- Television
- Travel
- Well
- Your Money

### Dataset Split

| Split | Number of Examples |
|---|---:|
| Training | 48,984 |
| Validation | 6,123 |
| Testing | 6,124 |
| **Total** | **61,231** |

Four duplicate records were removed from the original training split.

## Experimental Results

All four models were evaluated on the same held-out test set containing 6,124 examples.

### Final Test Performance

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Naive Bayes | 82.02% | 81.09% | 82.02% | 81.17% | 78.26% |
| Logistic Regression | 85.99% | 85.96% | 85.99% | 85.81% | 84.68% |
| **Linear SVM** | **87.12%** | **86.89%** | **87.12%** | **86.88%** | **86.00%** |
| BERT | 86.81% | 86.72% | 86.81% | 86.71% | 85.25% |

### Key Observations

- Linear SVM achieved the best performance across all five evaluation metrics.
- BERT achieved 86.81% test accuracy, making it a close second.
- Linear SVM outperformed BERT by 0.31 percentage points in accuracy.
- Logistic Regression performed better than Naive Bayes.
- Macro F1 was lower than weighted F1 for all models, indicating variation in class-level performance.

**Key finding:** Traditional TF-IDF-based Linear SVM remains highly competitive with a fine-tuned transformer for this particular news classification task.

## Project Structure

The exported project artifacts are organized as follows:

```text
N24News_Complete_Project/
│
├── metadata.json
│
├── Traditional_ML/
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── linear_svm.joblib
│   ├── tfidf_vectorizer.joblib
│   └── label_encoder.joblib
│
├── BERT/
│   ├── config.json
│   ├── model.safetensors
│   ├── training_args.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── label_encoder.joblib
│   └── Training_Checkpoints/
│
├── Datasets/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
└── Results/
    ├── model_comparison.csv
    ├── bert_validation.json
    ├── training_history.json
    ├── trainer_state.json
    └── final_test_comparison.csv
```

The Jupyter Notebook contains the data preparation, model training, evaluation, artifact export, and inference implementation.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Matplotlib
- Seaborn
- Google Colab
- ipywidgets

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install torch transformers datasets accelerate
pip install ipywidgets joblib
```

### 3. Open the notebook

Open the project notebook in Google Colab or Jupyter Notebook.

### 4. Configure dataset paths

Place the dataset files in the expected directory and update the notebook paths if necessary.

### 5. Execute the notebook

Run the notebook cells in sequence to:
- Load and inspect the dataset.
- Preprocess the text.
- Extract TF-IDF features.
- Train the traditional classifiers.
- Fine-tune BERT.
- Evaluate all models.
- Save trained model artifacts and comparison results.

**Note:** BERT fine-tuning requires substantial computational resources. A GPU-enabled environment such as Google Colab is recommended.

## Evaluation Metrics

The following metrics are used:

- **Accuracy:** Overall proportion of correctly classified articles.
- **Precision:** Proportion of predicted class assignments that are correct.
- **Recall:** Proportion of actual class examples correctly identified.
- **Weighted F1-score:** F1-score averaged using class support as weights.
- **Macro F1-score:** Unweighted average of F1-scores across all 24 categories.

Macro F1 is particularly useful for evaluating performance across categories with different numbers of training examples.

## Limitations and Future Work

- Evaluation is based on one fixed train-validation-test split.
- BERT processes a maximum of 384 tokens per article.
- Class distribution is imbalanced.
- Repeated-seed experiments and confidence intervals are not included.
- Controlled training-time and inference-latency comparisons have not been performed.

Potential future improvements include:
- Hyperparameter optimization.
- Class weighting and resampling.
- Per-category performance analysis.
- Experiments with different BERT sequence lengths.
- Efficiency and inference-time comparisons.

## Author

**Shreyansh Jain**  
M.Tech. Programme in Data Science  
Indian Institute of Technology Guwahati, Assam, India

## Acknowledgements

This project was developed as part of the M.Tech. Data Science academic coursework at the Indian Institute of Technology Guwahati.

---

**Note:** The reported results are specific to the dataset, preprocessing pipeline, model configurations, and experimental split used in this project.
