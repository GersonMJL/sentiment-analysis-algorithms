# PUBG Steam Reviews Sentiment Analysis

This project performs sentiment analysis on Steam reviews for PUBG: Battlegrounds using three different machine learning models: Decision Trees, Naive Bayes, and Support Vector Machines (SVM). The analysis is conducted in Jupyter Notebooks, and the workflow includes data collection, preprocessing, model training, evaluation, and comparison.

## Project Structure
- `get_data.py`: Script to fetch PUBG reviews from the Steam API and save them as CSV and Parquet files.
- `pubg-sentiment-analysis-dt.ipynb`: Sentiment analysis using Decision Trees.
- `pubg-sentiment-analysis-nb.ipynb`: Sentiment analysis using Naive Bayes.
- `pubg-sentiment-analysis-svm.ipynb`: Sentiment analysis using SVM.
- `requirements.txt`: List of required Python packages.

## Workflow Overview
1. **Data Collection**
   - Use `get_data.py` to fetch reviews for PUBG from the Steam API.
   - Reviews are saved in both CSV and Parquet formats for easy loading.

2. **Data Preprocessing**
   - Text is cleaned, tokenized, and stopwords are removed.
   - TF-IDF vectorization is used to convert text into features.

3. **Model Training & Evaluation**
   - Each notebook loads the data, preprocesses it, and trains a model (Decision Tree, Naive Bayes, or SVM).
   - Model performance is evaluated using accuracy and classification reports.
   - Example reviews are tested for sentiment prediction.

4. **Model Comparison**
   - The notebooks include markdown cells comparing the strengths and weaknesses of each model.

## How to Run
1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Fetch the data:
   ```bash
   python get_data.py
   ```
4. Open any of the notebooks in Jupyter or VS Code and run the cells to train and evaluate the models.

## Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies

## Notebooks
- **Decision Tree**: `pubg-sentiment-analysis-dt.ipynb`
- **Naive Bayes**: `pubg-sentiment-analysis-nb.ipynb`
- **SVM**: `pubg-sentiment-analysis-svm.ipynb`

Each notebook is self-contained and demonstrates the full pipeline for sentiment analysis using the respective model.

## License
This project is for educational and research purposes only.
