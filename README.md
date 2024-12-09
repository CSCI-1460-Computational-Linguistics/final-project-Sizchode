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

## **Implemented Equations**
The following equations, derived from the paper, are implemented in the model:

### Encoder-Decoder Framework
1. **Decoder's Output Probability**:
   ```math
   p(a|q) = \prod_{t=1}^{|a|}p(a_t | a_{<t}, q)
   ```

3. **Encoder's Output**:
   ```math
   h^L_k = \text{LSTM}(x_k, h^L_{k-1})
   ```

### Attention Mechanism
3. **Attention Scores**:
   ```math
   s^t_k = \frac{\exp(h^L_t \cdot h^L_k)}{\sum_{j=1}^{|q|} \exp(h^L_t \cdot h^L_j)}
   ```

4. **Context Vector**:
   ```math
   c^t = \sum_{k=1}^{|q|} s^t_k \cdot h^L_k
   ```

5. **Combined Representation**:
   ```math
   h^{att}_t = \tanh(W_1 h^L_t + W_2 c^t)
   ```

### Loss Function
6. **Objective Function**:
   ```math
   \mathcal{L} = -\sum_{(q,a) \in D} \log p(a|q)
   ```

---

## **Project Structure**
- **`Fall2024 Final Project.ipynb`**:
  Implementation for the seq2seq model.

- **`Video`**:
  Link: 
  

  
