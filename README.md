# DistilBERT Fine-Turning

## Architecture


### Dataset

IMDB 50k movie review. Train set is 70%, validation set is 10% and Test set is 20% of entire dataset split

#### With Training loop

Test accuracy: 93.82%

#### With HuggingFace Trainer

Test accuracy: 93.56%

### Results

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
-------

