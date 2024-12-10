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


### Sequence-to-Sequence Model(Section 3.1)

The encoder and decoder are two different L-layer recurrent neural
networks with long short-term memory (LSTM) units which recursively process tokens one by one.

1. **LSTM Representation** (Section 3.1, equation 2):
   Let $h_l∈ \mathbf{R_n}$ denote the hidden vector at time step t and layer l. $h^l_t$ is then computed by:
    \begin{align}
   h^l_t = \text{LSTM}(h^{l-1}_t, h^l_{t-1})
    \end{align}
   


2. **Model Output** (Section 3.1, equation 1):

  Seq2Seq maps natural language input $q = x_1 \dots x_{|q|}$ to a logical form representation of its meaning $a = y_1 \dots y_{|a|}$.
  $$p(a|q) = \prod_{t=1}^{|a|} p(y_t | y_{<t}, q)$$
   where $y_{<t} = y_1 \dots y_{t-1}$

3. **Predict t-th token output** (Section 3.1, equation 3)
$$p (y_t|y_{<t}, q) = softmax(W_oh^L_t)^T|e(y_t)$$
---

### Attention Mechanism (Section 3.3)
2. **Attention Scores** (Section 3.3, equation 5):
  $$s^t_k = \frac{\exp(h^L_t \cdot h^L_k)}{\sum_{j=1}^{|q|} \exp(h^L_t \cdot h^L_j)}$$
   Here, $s^t_k$ represents the attention score for the $k$-th encoder hidden state $h^L_k$ relative to the decoder hidden state $h^L_t$.

3. **Context Vector** (Section 3.3, equation 6):
  $$c^t = \sum_{k=1}^{|q|} s^t_k \cdot h^L_k$$
   The context vector$c^t$summarizes relevant information from the encoder hidden states $h^L_k$.

4. **Combined Representation** (Section 3.3, equation 7):
  $$h^{att}_t = \tanh(W_1 h^L_t + W_2 c^t)$$
   Here, $h^{att}_t$ is the combined representation of the decoder hidden state $h^L_t$ and the context vector $c^t$.

5. **Improved equation 3 using attention** (Section 3.3, equation 8)
$$p (y_t|y_{<t}, q) = softmax(W_oh^{att}_t)^T|e(y_t)$$

---

## **Project Structure**
- **`Fall2024 Final Project.ipynb`**:
  Implementation for the seq2seq model.

- **`Video`**:
  Link: 
  

  
