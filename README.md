
# Multimedia Sentiment Analysis

This project introduces a comprehensive system for sentiment analysis that incorporates multiple data types: text, images, and audio. Traditional sentiment analysis techniques primarily focus on text, limiting their effectiveness in situations where multimodal data is common. This project develops separate models for each data type—text, image, and audio—and combines their outputs to produce a unified sentiment score.

## Table of Contents
- [Introduction](#introduction)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
  - [Text Data](#text-data)
  - [Image Data](#image-data)
  - [Audio Data](#audio-data)
- [Model Development](#model-development)
  - [Text Sentiment Analysis](#text-sentiment-analysis)
  - [Image Sentiment Analysis](#image-sentiment-analysis)
  - [Audio Sentiment Analysis](#audio-sentiment-analysis)
- [Unified Sentiment Score Calculation](#unified-sentiment-score-calculation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Introduction
The project aims to overcome the limitations of traditional sentiment analysis methods by developing a system that can analyze sentiment across text, images, and audio. Using models designed specifically for each type of data and combining their results into a unified sentiment score, the approach provides a more accurate and detailed understanding of sentiment.

## Data Collection and Preprocessing

### Text Data
- **Dataset:** Sentiment140 dataset with 1.6 million tweets.
- **Preprocessing:**
  - Noise removal (URLs, mentions, hashtags).
  - Tokenization using NLTK.
  - Stop words removal.
  - Stemming and lemmatization.
  - Feature extraction using TF-IDF vectorization.

### Image Data
- **Dataset:** FER-2013 dataset with 35,887 grayscale facial images.
- **Preprocessing:**
  - Resizing to 48x48 pixels.
  - Normalization of pixel values.
  - Data augmentation (rotation, zoom, flipping).

### Audio Data
- **Dataset:** RAVDESS dataset with 7,356 emotional speech audio files.
- **Preprocessing:**
  - Noise reduction using the first second of the audio.
  - Feature extraction (MFCCs, chroma, spectral contrast).
  - Data augmentation (pitch shifting, time stretching).

## Model Development

### Text Sentiment Analysis
- **Model:** LSTM (Long Short-Term Memory) model.
- **Architecture:**
  - Embedding layer.
  - LSTM layers.
  - Dropout layers.
  - Dense layer with sigmoid activation.
- **Training:** Binary cross-entropy loss, Adam optimizer, early stopping.

### Image Sentiment Analysis
- **Model:** Hybrid model combining CNN (VGG16) and GRU.
- **Architecture:**
  - VGG16 base for feature extraction.
  - Time-distributed layer.
  - GRU layer for temporal relationships.
  - Dense layers with softmax activation.
- **Training:** Categorical cross-entropy loss, Adam optimizer, early stopping, and learning rate reduction on plateau.

### Audio Sentiment Analysis
- **Model:** CNN + Bidirectional GRU model.
- **Architecture:**
  - Conv1D layers for feature extraction.
  - Bidirectional GRU layers.
  - Dense layers with softmax activation.
- **Training:** Categorical cross-entropy loss, Adam optimizer, early stopping, and learning rate reduction on plateau.

## Unified Sentiment Score Calculation
Unified sentiment scores were calculated by averaging the predictions from each model to provide a comprehensive understanding of how sentiment is interpreted across different data types.

## Results
- **Text Sentiment Analysis:** 70.77% accuracy.
- **Image Sentiment Analysis:** 61.12% accuracy.
- **Audio Sentiment Analysis:** 81.94% accuracy.

## Conclusion
The project demonstrates that using a combination of text, images, and audio data is both practical and effective for sentiment analysis. This approach allows the model to predict sentiment more accurately than when relying on a single data type.

## Future Work
- Experiment with more diverse datasets to validate the model’s generalizability.
- Explore advanced architectures such as transformers for text and attention mechanisms for image and audio.
- Implement the model in a real-time application for dynamic sentiment analysis.

## References
- Lai, S., et al. (2021). Multimodal Sentiment Analysis: A Survey. *IEEE Transactions on Affective Computing*.
- Alam, M. H., et al. (2020). Sentiment Analysis using a Deep Ensemble Learning Model. *Journal of Information Science and Engineering*.
- Choube, A., & Soleymani, M. (2020). Punchline Detection using Context-Aware Hierarchical Multimodal Fusion. *Proceedings of the 28th ACM International Conference on Multimedia*.
