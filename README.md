# A3_DA6401_CE24D903
Transliteration and Text Generation Models
Overview
This project implements three distinct models for text processing using PyTorch, focusing on Hindi transliteration and character-level text generation. The models are designed to work with the Dakshina dataset for transliteration tasks and a sample text about the Indus Valley Civilization for text generation. The project includes:

Seq2Seq Model: A sequence-to-sequence model for transliterating Latin script to Devanagari script.
Seq2Seq Model with Attention: An enhanced Seq2Seq model with Bahdanau attention for improved transliteration performance.
LSTM with Transformer-Style Attention: An LSTM model with Transformer-style attention for character-level text generation, visualizing input-output character dependencies.

All models support hyperparameter tuning via Weights & Biases (WandB) sweeps, log detailed metrics, and include visualization capabilities. The transliteration models use the Dakshina dataset, while the text generation model processes a predefined text snippet.
Features

Dataset: 
Transliteration models use the Dakshina dataset (hi.translit.sampled.*.tsv) for training, validation, and testing.
Text generation model uses a sample text about the Indus Valley Civilization.


Models:
Seq2Seq: Encoder-decoder architecture with RNN, LSTM, or GRU cells, supporting bidirectional encoding.
Seq2Seq with Attention: Adds Bahdanau attention to focus on relevant input tokens during decoding.
LSTM with Transformer-Style Attention: Uses LSTM with scaled dot-product attention for next-character prediction.


Hyperparameter Tuning: Grid search over hyperparameters using WandB for transliteration models.
Metrics: Tracks loss, character accuracy, and word accuracy (for transliteration models).
Visualization:
Attention heatmaps for Seq2Seq with Attention, showing input-output dependencies.
Connectivity heatmap for LSTM model, visualizing attention weights for character prediction.


Logging: Integrates with WandB for logging metrics, model artifacts, sample predictions, and visualizations.
Error Handling: Saves model state on interruption (e.g., KeyboardInterrupt).
Output Storage: Saves test predictions and visualizations to disk.

Requirements

Python 3.8+
PyTorch
Pandas
NumPy
Weights & Biases (wandb)
Tabulate
Matplotlib
Seaborn

Install dependencies using:
pip install torch pandas numpy wandb tabulate matplotlib seaborn

Dataset

Transliteration Models: Use the Dakshina dataset for Hindi transliteration. Place the files (hi.translit.sampled.train.tsv, hi.translit.sampled.dev.tsv, hi.translit.sampled.test.tsv) in /home/user/Downloads/dakshina_dataset_v1.0/hi/lexicons/. Each file is a tab-separated file with two columns: Devanagari text (target) and Latin text (source).
Text Generation Model: Uses a hardcoded text snippet about the Indus Valley Civilization, included in the script.

Project Structure
The project consists of three scripts, each with specific components:
1. Seq2Seq Model (transliteration_seq2seq.py)

TransliterationDataset: Custom PyTorch Dataset class to load and preprocess text data, build vocabularies, and convert text to indices.
Encoder: RNN-based encoder (supports RNN, LSTM, GRU) with optional bidirectional processing.
Decoder: RNN-based decoder for generating target sequences.
Seq2Seq: Combines encoder and decoder, handles teacher forcing, and supports inference.
Training/Evaluation: Functions for training, evaluating, and calculating accuracies.
WandB Integration: Logs metrics and model artifacts.
Sweep Configuration: Grid search over hyperparameters like embedding size, hidden dimensions, and optimizer.

2. Seq2Seq Model with Attention (transliteration_seq2seq_attention.py)

TransliterationDataset: Same as above, with shared vocabulary across train/dev/test sets.
Attention: Implements Bahdanau attention to compute weights for encoder outputs.
Encoder: Returns encoder outputs and hidden/cell states.
Decoder: Incorporates attention to generate context vectors for decoding.
Seq2Seq: Manages encoder-decoder pipeline with attention, returns attention weights.
Visualization: Generates attention heatmaps for up to 9 sample predictions using Matplotlib and Seaborn.
WandB Integration: Logs metrics, model artifacts, sample predictions, and attention heatmaps.
Output Storage: Saves test predictions to predictions_attention/test_predictions.csv.
Sweep Configuration: Includes additional hyperparameters like beam size and teacher forcing ratio.

3. LSTM with Transformer-Style Attention (text_generation_lstm_attention.py)

Data Preparation: Preprocesses a sample text, creating sequences of fixed length for next-character prediction.
LSTMWithAttention: Combines LSTM with scaled dot-product attention (Transformer-style) to predict the next character.
Training: Trains the model with Adam optimizer, cross-entropy loss, and learning rate scheduling.
Visualization: Generates a connectivity heatmap showing attention weights between input and output characters.
Output Storage: Saves the heatmap to connectivity_heatmap_1.png.

Usage
Each script must be run separately to execute its respective task. Ensure the required dataset is prepared for the transliteration models.

Setup WandB (for transliteration models):

Ensure you have a WandB account and API key.

Log in to WandB:
wandb login




Prepare Dataset (for transliteration models):

Download the Dakshina dataset and place the .tsv files in /home/user/Downloads/dakshina_dataset_v1.0/hi/lexicons/.
The text generation model does not require external data, as it uses a hardcoded text snippet.


Run the Scripts:

Seq2Seq Model:
python train_Q1.ipynb


Runs a grid search over hyperparameters using train_sweep(), logging results to WandB.
Saves the best model as best_model_<run_id>.pt and the last model on interruption as last_model_<run_id>.pt.


Seq2Seq Model with Attention:
python train_Q2.ipynb


Runs a grid search using train_sweep(), logging metrics, sample predictions, and attention heatmaps to WandB.
Saves test predictions to predictions_attention/test_predictions.csv and attention heatmaps to predictions_attention/attention_heatmaps.png.
Saves models as above.


LSTM with Transformer-Style Attention:
python Q6_DA6401.ipynb


Trains an LSTM model with Transformer-style attention on the sample text.
Generates and saves a connectivity heatmap to connectivity_heatmap_1.png.
No WandB integration; runs as a standalone script.




Monitor Results:

For transliteration models, view training progress, metrics, sample predictions, and attention heatmaps (for the attention model) in the WandB dashboard.
For the text generation model, check the console output for training progress and the saved heatmap file for visualization.



Hyperparameters
Seq2Seq Model

Embedding Dimension, Hidden Dimension, Encoder/Decoder Layers, Cell Type (RNN, LSTM, GRU), Dropout, Learning Rate, Batch Size, Optimizer (Adam, AdamW, RMSprop), Gradient Clipping, Weight Decay, Bidirectional.
Sweep runs for 100 configurations.

Seq2Seq Model with Attention

Same as above, plus Beam Size (1, 3, 5) and Teacher Forcing Ratio (0.3, 0.5, 0.7).
Sweep runs for 50 configurations.

LSTM with Transformer-Style Attention

Fixed hyperparameters: Input Dimension (1), Hidden Dimension (256), Sequence Length (20), Batch Size (32), Temperature (0.3), Epochs (300), Learning Rate (0.001 with StepLR scheduler).

Model Architecture
Seq2Seq Model

Encoder: RNN-based, processes Latin script input, outputs hidden/cell states.
Decoder: Generates Devanagari script output, uses teacher forcing during training.
Seq2Seq: Combines encoder and decoder, supports bidirectional encoding.

Seq2Seq Model with Attention

Attention: Bahdanau attention computes weights for encoder outputs.
Encoder: Returns outputs and hidden/cell states.
Decoder: Uses attention to create context vectors for decoding.
Seq2Seq: Manages encoder-decoder pipeline, returns attention weights.

LSTM with Transformer-Style Attention

LSTMWithAttention: Combines LSTM with scaled dot-product attention to predict the next character.
Attention: Uses query, key, and value projections with temperature scaling for sharper attention.
Output: Predicts the next character based on the context vector from the last sequence position.

Metrics

Transliteration Models:
Character Accuracy: Percentage of correctly predicted characters (excluding <sos>, <eos>, <pad>).
Word Accuracy: Percentage of correctly predicted words (entire sequence must match exactly).
Loss: Cross-entropy loss, ignoring padding tokens.


Text Generation Model:
Loss: Cross-entropy loss for next-character prediction.



Visualizations

Seq2Seq with Attention: Attention heatmaps for up to 9 sample predictions, showing input-output dependencies, saved to predictions_attention/attention_heatmaps.png and logged to WandB.
LSTM with Transformer-Style Attention: Connectivity heatmap showing attention weights between input and output characters, saved to connectivity_heatmap_1.png.

Sample Predictions (Transliteration Models)

Logs a table to WandB with:
Latin input
Devanagari target
Devanagari predicted
Correctness (✅ or ❌)

