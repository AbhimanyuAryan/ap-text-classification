# Introduction


Nowadays high-performing natural language models gained much relevance. In this project, we used three different pre-trained LLMs in order to classify opinions as positive or negative. The LLMs utilized in our study include:

- Distilbert
- Mistral
- Gemma

Our [GitHub repository](https://github.com/AbhimanyuAryan/llm-finetuning/) contains all the code and explanations for every decision we have made.

# DistilBERT Fine-Tuning

### Architecture


### Fine Tuning

**Dataset**: IMDB 50k movie review. Train set is 70%, validation set is 10% and Test set is 20% of entire dataset split.

**Training**: Summarize important aspects of the code.

1. Learning Rate: Graphs show a decreased learning rate, it is common in
training as it shows model is converging by doing smaller updates to the
weights as training progresses
2. Global Step: This indicates the number of batches the has been trained
on. It increase with each batch processed
3. Gradient Norm: A slight decrease in gradient norm suggests that the
updates to the model’s weights are becoming more stable as training goes
on
4. Epoch: It just means that the model is seeing the data repeatedly, and
with each pass(epoch), it’s learning more about dataset
5. Loss: The declining loss means model’s prediction are getting closer to
the actual labels, which the model is learning effectively.

**Train loop**:

<div align="center">

![train_loop](images/distilbert/wandb_graphs_train_loop.png)

</div>

**Test loop**:

<div align="center">

![trainer](images/distilbert/wandb_hf_trainer.png)

</div>

### Results


|       Method            | Accuracy |
|:-----------------------:|:--------:|
| Training loop           | 93.82%   |
| HuggingFace Trainer     | 93.56%   |

# Mistral Fine-Tuning

### Introduction

Mistral is an [open-source model](https://github.com/mistralai/mistral-src) owned by the company [Mistral AI](https://mistral.ai/). It was published with the a [paper](https://arxiv.org/abs/2310.06825) and it is famous due to its performance and efficiency. Compared to the Llama model, Mistral surpasses the first version of Llama in all evaluated benchmarks. With the second version of Llama, Mistral is better in mathematics and code generation. 

We build a notebook based on a [public notebook](https://www.kaggle.com/code/lucamassaron/fine-tune-mistral-v0-2-for-sentiment-analysis). In relation with Mistral version, we used the version [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

### Architecture

Mistral is based on Transformers Architecture.

The parameters of Mistral architecture are:

<div align="center">

| Parameter    | Value  | Explanation |
|:------------:|:------:|-------------|
| dim          | 4096   | Dimensionality of the model's embeddings and hidden states. This defines the size of the vectors used throughout the model.                                                    |
| n_layers     | 32     | Number of transformer blocks in the model. |
| head_dim     | 128    | Dimensionality of each attention head. |
| hidden_dim   | 14336  | Dimensionality of the feed-forward layer within each transformer block. |
| n_heads      | 32     | Number of attention heads in the multi-head attention mechanism.|
| n_kv_heads   | 8      | Number of key-value heads used in attention. |
| window_size  | 4096   | Size of the local context window used in models with attention mechanisms that restrict the range of attention to a local context. |
| context_len  | 8192   | Maximum length of the input sequences. |
| vocab_size   | 32000  | How many unique tokens (words, subwords, or characters) the model can represent. |


</div>

**Sliding Window Attention (SWA)** The sliding window attention pattern employs a fixed-size window attention surrounding each token. This means that each position in a layer can attend to hidden states from the previous layer within a range of 4096 tokens behind it and up to itself.
<div align="center">
    
![swa](images/mistral/SWA.png)

</div>

**Rolling Buffer Cache**: Fixed cache size.
<div align="center">
    
![fixed_cache](images/mistral/Cache.png)

</div>

**Pre-fill and Chunking**: Devide the prompt into smaller pieces and then work with those pieces instead of the all prompt. 
<div align="center">
    
![prefillchunking](images/mistral/PreFillChunking.png)

</div>


### Fine Tuning

**Dataset**: IMDB movie review, 900 cases for training, 100 cases for evaluating and 2500 cases for testing. The data is balanced.

**Training** For tune Mistral we have used the library Supervised Fine-tuning Trainer, as known as [SFTT](https://huggingface.co/docs/trl/sft_trainer) instead of the normal [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

- More appropriated to text classification problems.
- Our dataset is not that large.
- The training process is faster.
- Uses less memory.

<div align="center">


| Epoch | Training Loss | Validation Loss |
|:-----:|:-------------:|:---------------:|
|   1   |    2.019900   |    2.116099     |
|   2   |    1.980300   |    2.124171     |
|   3   |    1.883400   |    2.153847     |
|   4   |    1.827500   |    2.167378     |

</div>

<div align="center">
    
![train_loss_values](images/mistral/TrainLoss.png)
![eval_loss_values](images/mistral/EvalLoss.png)


</div>

### Results

<div align="center">
    
| Stage    | Metric                        | Value |
|:--------:|:-----------------------------:|:-----:|
| Original | Accuracy                      | 63.0% |
| Original | Accuracy for negative reviews | 98.0% |
| Original | Accuracy for positive reviews | 28.0% |
| Tuned    | Accuracy                      | 96.0% |
| Tuned    | Accuracy for negative reviews | 97.4% |
| Tuned    | Accuracy for positive reviews | 94.6% |


</div>

# Gemma Zero Shot Learning

The Gemma directory contains four Jupyter notebooks used to explore the Gemma model's text classification capabilities.

Initially, we aimed to fine-tune this model, just like we did with *Mistral* and *Distilbert*. Sadly, that was not feasible because every attempt ended up consuming all available resources in both Kaggle and Google Colab.

So, we focused on implementing Zero Shot Learning to test the model's performance in this task. We successfully managed to implement this type of classification. After doing so, we attempted to implement both One Shot classification and Few Shot classification.

These attempts led to failure, as once again, we consumed all the resources.

However, we still want to provide the code for all our attempts. We will not go into as much detail as we did for the other two models, since we can consider this to be an extra that improves the overall work.

We kept some code blocks that are not particularly relevant, such as some pip installs that ended up not being utilized in the final version. Yet, these would be necessary for the failed fine-tuning. We also kept under comments a function that can be used to clear resources on Google Colab.

Three of our notebooks implement Zero Shot Learning. These notebooks contain the exact same content, but we created them to prove the randomness in zero-shot learning as results vary on each execution. Then we have a notebook that contains the same basic code with the code needed for One Shot and Few Shot classification added to it. We kept the output that showcases the error.

The final not we consider important to make is that the following code is used to access the imdb dataset. We could have used the datasets package, to alude to it we kept the command to install this package on the notebooks.

But this ends up becoming not much relevant because as long as the dataset contains a column with text to classify and a column with either 0 or 1 most of the notebook remains usable.

### Architecture

To do

### Zero Shot Learning

Summarize the code here.

### Results

As for the results obtained with zero shot learning for this model we can say that as mentioned they are not that consistent and a bit random, as they  do not remain the same over the
iterations, which proves randomness, also the results were as expected a lot worse than the fine tuned examples which proves just how important it is to properly train these models.
Still, it is interesting to analyse how good the model’s base form is at this task as we managed to get scores that were near or above 50% for 2500 cases.
This is in our eyes quite remarkable and we assume that the reviews that were wrongly classified could be more complex ones with nuanced contexts

The results obtained can be consulted on this table:

<div align="center">

| Execution | Accuracy | Recall | Precision |
|:---------:|:--------:|:------:|:---------:|
|    #1     |   0.49   |  0.49  |    0.48   |
|    #2     |   0.62   |  0.62  |    0.62   |
|    #3     |   0.60   |  0.60  |    0.71   |

</div>

# Conclusions

Unsurprisingly, Mistral was the best in terms of performance. This outcome was anticipated, given Mistral's superior pre-training compared to Distilbert, and the absence of specific tuning for Gemma.

Our journey has been truly fulfilling, marked by the exploration of diverse models and methodologies, each offering unique insights into the realm of natural language processing.

---

# References

1. [Mistral 7B, Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed, 2023. arXiv eprint: 2310.06825, primary class: cs.CL.](https://arxiv.org/abs/2310.06825)
