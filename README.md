# Sentiment Analysis - Indodax

**Author:** Alessandro Javva Ananda Satriyo

This project focuses on performing sentiment analysis on data related to Indodax, the largest cryptocurrency exchange in Indonesia. The primary goal is to classify sentiments (positive, neutral, negative) from user reviews of the Indodax app, using various machine learning techniques.

## Project Overview

The notebook (`sentiment-analysis.ipynb`) contains several key sections:

1. **Data Extraction**: Data is extracted using the `google-play-scraper` library, focusing on user reviews from the Google Play Store. This section handles gathering relevant data for sentiment analysis.
2. **Data Preprocessing**: Includes steps such as text cleaning, tokenization, and encoding of sentiment labels to prepare the raw data for analysis.
3. **Feature Extraction**:
   - **TF-IDF**: Converts text data into numerical features for machine learning models.
   - **Bag of Words (BoW)**: Another method used for converting text data into features, particularly combined with the Simple RNN model.
4. **Model Training and Evaluation**: Implements and evaluates several machine learning models:
   - **LSTM** and **MLP** models using TF-IDF.
   - **Simple RNN** model using Bag of Words for feature extraction.
5. **Inference and Output**: Performs inference on test data and produces categorical sentiment labels (e.g., positive, neutral, negative).

## Indodax Overview

Indodax is the largest cryptocurrency exchange in Indonesia. It allows users to trade various digital assets such as Bitcoin, Ethereum, and others. The platform is known for its secure trading environment and is widely used by crypto enthusiasts and investors in Indonesia.

## Research Purpose

This project is purely for research purposes. It aims to explore sentiment analysis techniques and their applicability to real-world data related to Indodax. The models and findings presented in this notebook are not intended for commercial or production use.

## Project Prerequisites

### Required Libraries

To replicate the analysis, you need the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `google-play-scraper`

You can install the necessary libraries using pip:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib google-play-scraper
```

### Running the Notebook

1. Clone the repository to your local machine.
2. Ensure you have all the prerequisites installed.
3. Open the notebook using Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook sentiment-analysis.ipynb
   ```

4. Execute the cells in order to reproduce the results.

## Results

The notebook concludes with an evaluation of the machine learning models, providing accuracy metrics for both validation and test datasets. Additionally, the notebook shows examples of sentiment predictions for sample inputs.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
