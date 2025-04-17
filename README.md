PART 1

The goal is to classify the sentiment of economic news posts as either “good” or “bad” news. We completed the provided model for this task and explored key steps in handling such data, including normalization (# added). We also compare results with and without this normalization to assess its impact on performance.

Exercise 1: Complete the following code to construct the RNN classifier

We updated the forward method. We first pass the input tensor x (which contains sequences of word embeddings) through the LSTM layer using self.rnn(x), obtaining r_out, which contains the hidden states for each timestep in the sequences. Since each input sequence may have a different actual length (ignoring padding or junk tokens), we need to extract the relevant hidden state that corresponds to the last real word in each sequence. This is done by iterating over the batch and using r_out[d, lengths[d]-1, :] to select the hidden state at the correct final timestep for each sequence. These selected hidden states are then stacked into a tensor aux of shape (batch_size, hidden_dim). We apply dropout to aux for regularization, pass it through a fully connected layer fc1 to project it into the output space (e.g., number of classes), and then apply a logsoftmax to produce log-probabilities for classification. This entire process ensures that the model classifies each sequence based on the meaningful content only, ignoring any padding.

Exercise 2: evaluate the RNN output for ‘x_input’ and check that the output dimensions make sense.

We prepare a batch of input data and pass it through an RNN model. First, x_input, which contains a tensor of input sequences (typically word embeddings for each token in the batch), is assigned to x_batch for clarity. Then, the list length is updated to include only the sequence lengths corresponding to the current batch indices (idx), ensuring the model knows the actual (non-padded) lengths of each input. Finally, the forward method of the RNN (my_RNN.forward) is called with the input batch and their respective lengths. 
This allows the model to process the batch correctly, particularly when dealing with sequences of varying lengths, and returns the output o, which generally contains the model's predictions for the batch.

Exercise 3: Complete the following class, which inherits the previous one and it adds a training loop, an evaluation method, and functionalities to save the model every few epochs

We correctly filled the training and validation batches to properly construct x_input using list comprehensions over docs_train and docs_val, and the sequence lengths are accurately extracted using the corresponding len_train and len_val lists. 

Exercise 4: Instantiate and train the class using a hidden state of 20 dimensions and dropout probability equal to 0.3. Train for 40 epochs (can take a while) & Exercise 5: Plot both validation and training loss. Recover the model parameters for the epoch that minimized the validation loss.

After using a hidden state of 20 dimensions and dropout with a probability of 0.3 through 40 epochs, we have plotted the validation and training loss obtaining this graph:

Here we can observe how the training loss decreases smoothly at the same time that the validation curve slightly decreases but will not lower 0.2. SEE GRAPH 1. Visually, we can assume there is some overfitting due to the perfect fit of training but worse results for validation set.

Exercise 6: Using the method predict_proba, compute the accuracy and class probabilities for the data in the test set. Note that the method returns log-probabilities that you have to exponentiate.

We obtained an accuracy of 0.93 (rounded) for the test set which is a very good result 
For the probabilities, we start by finding the best model, which is determined by identifying the epoch that resulted in the lowest validation loss during training. Once this optimal epoch is found, we load the corresponding saved model parameters to ensure we're evaluating the most effective version of the model. Next, we evaluate this best model on the test dataset to obtain final performance metrics, giving us a clear idea of how well it generalizes to unseen data. To put these results in context, we also compute the baseline accuracy by checking how well a naive model that always predicts the majority class would perform. Finally, to gain a more intuitive understanding of the model's behavior, we display several random test examples along with their true labels and the model’s predictions, providing insight into specific cases where the model succeeds or struggles.

•	Best Model Found: Epoch 19 with Validation Loss: 0.2
•	Test Accuracy (Best Model): 0.9344
•	Baseline Accuracy (Majority Class): 0.8775
•	Number of class 0 (Neutral/Positive): 1003
•	Number of class 1 (Negative): 140

The dataset is heavily imbalanced toward Neutral/Positive examples, which explains why random samples all show this class. While 93% accuracy is good, the high baseline means the model's improvement is more modest than it might first appear. The model seems very confident in its predictions (near 100% confidence in some cases), which may indicate it's well-calibrated for this task. SEE TABLE 1

Lastly, for our ROC curve we see it increases to achieve the 1.0 and gets just by it. This gives a value of AUC ROC for LSTM is 0.91. SEE GRAPH 2.


OPTINAL PART

MLP Approach Overview
In this optional part of the project, a Multi-Layer Perceptron (MLP) was implemented to classify sentiment in financial news. Instead of processing word sequences like an RNN, the MLP takes a simpler approach by using the average of all word embeddings in each text as a single input vector. The model architecture consisted of three layers with 10 and 5 hidden units, and it was trained for 40 epochs using the Adam optimizer.

Performance Results: SEE GRAPH 3
The MLP achieved a test accuracy of 87.76%, nearly identical to the 87.75% baseline (majority class prediction). However, its AUC-ROC score was 0.57, indicating very limited ability to distinguish between classes which is essentially close to random guessing.

Insights and Interpretation
Although the MLP's accuracy seems high, its similarity to the baseline and poor AUC-ROC show that it struggles to capture meaningful patterns beyond the dominant class. It likely predicts "Neutral/Positive" for most inputs, making it unreliable for minority class identification.
Why the MLP Falls Short? Averaging word embeddings causes loss of sequential information. As a result, the MLP cannot detect context, negation, or word order—key elements in sentiment analysis. This flattening of text data limits the model’s ability to distinguish subtle emotional cues.

Comparing to RNN
The RNN model, on the other hand, achieved around 93% accuracy, demonstrating its superior performance. Its sequential nature allows it to retain contextual meaning, which is critical in understanding sentiment. The RNN's ~5.25 percentage point improvement over both the MLP and the baseline confirms the effectiveness of modeling word order and structure.

PART 2 - RNNs for Sentiment Analysis (with Attention)

We've added an attention mechanism to the RNN model which:
•	Uses the last hidden state (h_ℓ) as the query
•	Uses all hidden states (h_0 to h_ℓ) as keys
•	Implements a two-layer MLP for the attention mechanism:
o	First layer: Combines query and key information with a tanh activation
o	Second layer: Produces unnormalized attention weights
•	Applies a softmax to get normalized attention weights (α)
•	Computes the context vector (c) as a weighted sum of hidden states using the attention weights

The notebook includes visualization of the training and validation loss curves, ROC curve for model evaluation and attention weights for specific examples to see which words the model focuses on.

Masking
A crucial part of the implementation is masking the garbage tokens (#) that were added to equalize the sequence lengths:
•	 we created a mask based on the true lengths of each sequence
•	Applied the mask to the attention scores by setting the scores for garbage tokens to negative infinity before the softmax
•	This ensures that the attention weights are only calculated for the actual tokens in each sequence

The model trains for 40 epochs, showing the training and validation loss every 5 epochs and it identifies the best model based on validation loss. Then, it evaluates the model on the test set and compute performance metrics Finally, we visualize attention weights for several correctly and incorrectly classified examples.

 
As we can see there is less overfitting compared to PART 1. (SEE GRAPH 4 and GRAPH 5)

The attention mechanism offers two main benefits:
1.	improved performance: By focusing on the most important words, the model may higher accuracy and AUC (0.9388 and 0.9119 respectively)
2.	Interpretability: The attention weights show which parts of the input the model considers most important for its decision.

Even if the performance doesn't significantly improve, the interpretability provided by the attention mechanism is valuable, as it helps us understand how the model is making its decisions. See Annex Part 2 to see which words in each sentence had the highest influence on the model's prediction.

Comparison with original RNN model:
Attention RNN - Test Accuracy: 0.9388, AUC: 0.9119
Original RNN - Test Accuracy: 0.9396, AUC: 0.9309

Conclusion:
The attention mechanism has improved the model's performance.
The attention weights show which words in the sentence the model focuses on when making the prediction


Final Conclusion:

In this project, we aimed to classify the sentiment of financial news as "good" or "bad." We started with an RNN model, achieving a test accuracy of 93% while handling varying sequence lengths with padding. However, the dataset’s imbalance toward "Neutral/Positive" news limited the improvement, as the baseline accuracy was 87.75%.

In the optional part, we implemented an MLP model, which averaged word embeddings. While it achieved an accuracy of 87.76%, it performed poorly in distinguishing classes due to the loss of sequential information, making it less effective than the RNN.

We then added an attention mechanism to the RNN, which slightly improved accuracy (93.88%) and AUC (0.9119). The attention model also offered interpretability by showing which words influenced predictions, making the model more transparent.

In summary, the RNN with attention outperformed the MLP and provided valuable insights through attention weights. This project highlights the importance of sequence modeling and attention mechanisms in sentiment analysis of financial news.
