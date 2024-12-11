# **Seq2Seq with Attention for Language-to-Logical Form Parsing**

## **Overview**
This project implements a sequence-to-sequence (Seq2Seq) model with attention to convert natural language utterances into machine-readable logical forms. Inspired by the paper [Language to Logical Form with Neural Attention](https://aclanthology.org/P16-1004.pdf), the model employs an encoder-decoder framework augmented with an attention mechanism to improve logical form generation.

The main objective is to maximize the likelihood of generating the correct logical form \( a \) given a natural language input \( q \). The training dataset consists of paired examples of natural language queries and their corresponding logical forms.

---

## **Features**
- **Encoder-Decoder Framework**:
  - A 2-layer stacked LSTM encoder processes input queries into a sequence of hidden states.
  - A decoder, initialized with the encoder's final hidden and cell states, generates the logical form tokens step-by-step.
  
- **Attention Mechanism**:
  - Implements attention to compute a weighted context vector from encoder hidden states for each decoder step.

- **Loss Function**:
  - Uses negative log-likelihood loss (`NLLLoss`) with padding tokens ignored during computation.

- **Optimization**:
  - Trains using the RMSProp optimizer, as recommended in the paper.

- **Greedy Decoding**:
  - Implements token-by-token greedy decoding for inference.

---

## **Project Structure**
- **`Fall2024 Final Project.ipynb`**:
  Implementation for the seq2seq model with attention.

- **`Video`**:
  Link: https://drive.google.com/file/d/1lb5B3TbgnI-m7AItfyvncDK07eXEfilC/view?usp=drive_link

- **`Contact`**:
  Zhenke Liu(zliu328@brown.edu)
  

  
