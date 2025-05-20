# A3_DA6401_CE24D903
**This is for the question 1**
Dependencies
Python 3.8+
torch, pandas, numpy, matplotlib, seaborn, wandb, tabulate
**pip install torch pandas numpy matplotlib seaborn wandb tabulate**

Setup
Save the code as train_Q1.ipynb
Update dataset paths in train_sweep function if needed.
Usage
Training
Run hyperparameter sweep:
python train_Q1.ipynb
Performs a grid search over hyperparameters (embedding size, hidden size, cell type, etc.).
Trains up to 100 models (configurable via count in wandb.agent).
Saves the best model as best_model_<run_id>.pt based on validation word accuracy.
Logs metrics (loss, character accuracy, word accuracy) and sample predictions to WandB

**This is for the question 2**
Setup
Save the code as train_Q2.py.
Place best_model_<run_id>.pt in the script directory (if using a saved model).
Update dataset paths in the script.
**Usage**
Training
Run hyperparameter sweep:
python train_Q2.ipynb
Trains up to 50 models, saves best as best_model_<run_id>.pt.
Logs metrics to WandB.
Saves heatmaps (predictions_attention/attention_heatmaps.png) and predictions (predictions_attention/test_predictions.csv).

**This is for the Question Number 6**
**LSTM with Attention on Character-Level Text Prediction**
This project implements a character-level text prediction model using an LSTM enhanced with Transformer-style attention. The model is trained on a historical text excerpt to predict the next character in a sequence, and attention heatmaps are used to visualize the model's focus across input sequences.

📌 **Overview**
Model Architecture: LSTM + Transformer-style attention mechanism

Task: Predict next character in a text sequence

Visualization: Heatmap of attention weights to understand character dependencies

Use Case: Educational demonstration of attention mechanisms in sequence modeling

🧠 **Key Features**
Custom Attention Layer: Computes context vectors via scaled dot-product attention over LSTM outputs.

Sharpened Focus: Attention temperature scaling allows better control over focus sharpness.

Connectivity Visualization: Attention weights plotted as a heatmap to analyze model behavior.

Character-Level Processing: Offers fine-grained insights into how characters influence predictions.

🧰 **Requirements**
Ensure the following libraries are installed:

pip install numpy torch matplotlib seaborn
🚀 **How to Run**
Clone or download the script.

Run the Python file:

python Q6_DA6401.ipynb
After training (300 epochs), the script:

Outputs attention weights for a test input.

Generates a heatmap (connectivity_heatmap_1.png) showing input-output character dependencies.

📊 **Output**
Heatmap File: connectivity_heatmap_1.png (saved in the script's directory)

Logs: Epoch-wise training loss and attention summaries to understand focus alignment.
When decoding output character 6 ('r'), the model focuses most on input character 16 ('a') with weight 0.1423
🧩 **Model Details**
Architecture
text
Copy
Edit
Input (seq_len x 1)
     ↓
LSTM Layer (256 hidden units)
     ↓
Self-Attention Mechanism
     ↓
Fully Connected Layer → Softmax Output
Parameters
seq_length: 20 (input sequence length)

hidden_dim: 256 (LSTM hidden units)

temperature: 0.3 (attention sharpness)

epochs: 300

📖 **Training Data**
The text is a historical narrative describing the discovery of the Indus Valley Civilization, preprocessed to lowercase characters with spacing normalized.

In the same file there is code for the **multi head attention** the working remains the same, only visualization will change, it will have Log-scaled attention heatmap 
