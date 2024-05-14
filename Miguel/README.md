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