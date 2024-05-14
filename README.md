# DistilBERT Fine-Tuning

### Architecture


### Dataset

IMDB 50k movie review. Train set is 70%, validation set is 10% and Test set is 20% of entire dataset split

### Training

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

#### Train loop

![train_loop](images/distilbert/wandb_graphs_train_loop.png)

#### Test loop

![trainer](images/distilbert/wandb_hf_trainer.png)

### Results

- Final accuracy with Training loop: 93.82%
- Final accuracy with HuggingFace Trainer: 93.56%

-----

# Mistral Fine-Tuning

### Architecture

Mistral is based on Transformers Architecture.

The parameters of Mistral architecture are:

| Parameter    | Value   |
|--------------|---------|
| dim          | 4096    |
| n_layers     | 32      |
| head_dim     | 128     |
| hidden_dim   | 14336   |
| n_heads      | 32      |
| n_kv_heads   | 8       |
| window_size  | 4096    |
| context_len  | 8192    |
| vocab_size   | 32000   |


**Sliding Window Attention (SWA)** The sliding window attention pattern employs a fixed-size window attention surrounding each token. This means that each position in a layer can attend to hidden states from the previous layer within a range of 4096 tokens behind it and up to itself.

![swa](images/mistral/SWA.png)


**Rolling Buffer Cache**: Fixed cache size.

![fixed_cache](images/mistral/Cache.png)


**Pre-fill and Chunking**: Devide the prompt into smaller pieces and then work with those pieces instead of the all prompt. 

![prefillchunking](images/mistral/PreFillChunking.png)

### Dataset

IMDB movie review, 900 cases for training, 100 cases for evaluating and 2500 cases for testing. The data is balanced.


### Training

The loss function that we have used was Cross-Entropy Loss.

LossFunction(p, q) = -$\sum_{x=1}^{N} p(x) * log(q(x))$

Where:
- p is the true probability distribution (usually a one-hot encoded vector representing the ground truth).
- q is the predicted probability distribution (the output probabilities from the model).
- x represents each class or category in the distribution.

![loss_values](images/mistral/LossValues.png)

### Results

- Final Accuracy with Supervised Fine-tuning Trainer: 96.40%


# Experimenting Zero Shot Learning for the Gemma Model

This directory contains four Jupyter notebooks used to explore the Gemma model's text classification capabilities.

Initially, we aimed to fine-tune this model, just like we did with *Mistral* and *Distilbert*. Sadly, that was not feasible because every attempt ended up consuming all available resources in both Kaggle and Google Colab.

So, we focused on implementing Zero Shot Learning to test the model's performance in this task. We successfully managed to implement this type of classification. After doing so, we attempted to implement both One Shot classification and Few Shot classification.

These attempts led to failure, as once again, we consumed all the resources.

However, we still want to provide the code for all our attempts. We will not go into as much detail as we did for the other two models, since we can consider this to be an extra that improves the overall work.

We kept some code blocks that are not particularly relevant, such as some pip installs that ended up not being utilized in the final version. Yet, these would be necessary for the failed fine-tuning. We also kept under comments a function that can be used to clear resources on Google Colab.

Three of our notebooks implement Zero Shot Learning. These notebooks contain the exact same content, but we created them to prove the randomness in zero-shot learning as results vary on each execution. Then we have a notebook that contains the same basic code with the code needed for One Shot and Few Shot classification added to it. We kept the output that showcases the error.

The final not we consider important to make is that the following code is used to access the imdb dataset. We could have used the datasets package, to alude to it we kept the command to install this package on the notebooks.

```python
def save_data():
    # URL of the dataset to download
    url = "https://github.com/rasbt/python-machine-learning-book-3rd-edition/raw/master/ch08/movie_data.csv.gz"
    # Extract filename from URL
    filename = url.split("/")[-1]

    # Download and save the dataset
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    # Extract the gzip file
    with gzip.open(filename, 'rb') as f_in:
        with open('movie_data.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Define a function to prepare test data for classification
def prepare_data_test(df):
    # Extract a subset of test data from the DataFrame
    X_test = df.iloc[40000:42500]
    # Assign 'review' column to 'text' for convenience
    X_test['text'] = X_test['review']
    # Ensure 'text' column contains string data
    X_test['text'] = X_test['text'].astype(str)
    # Extract true labels for the test data
    y_true = list(X_test['sentiment'])
    return X_test.drop(['sentiment','review'],axis=1), y_true

# Define a function to prepare data for classification
def prepare_data():
    # Read the movie review dataset into a DataFrame
    df = pd.read_csv('movie_data.csv')
    # Prepare test data using the previously defined function
    return prepare_data_test(df)

# Download and save the movie review dataset
save_data()
# Prepare test data for classification
X_test, y_true = prepare_data()
```

But this ends up becoming not much relevant because as long as the dataset contains a column with text to classify and a column with either 0 or 1 most of the notebook remains usable.

### Results

As for the results obtained with zero shot learning for this model we can say that as mentioned they are not that consistent and a bit random, as they  do not remain the same over the
iterations, which proves randomness, also the results were as expected a lot worse than the fine tuned examples which proves just how important it is to properly train these models.
Still, it is interesting to analyse how good the model’s base form is at this task as we managed to get scores that were near or above 50% for 2500 cases.
This is in our eyes quite remarkable and we assume that the reviews that were wrongly classified could be more complex ones with nuanced contexts
