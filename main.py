from datasets import load_dataset
from setfit import SetFitModel, Trainer
from sklearn.preprocessing import LabelEncoder

# sentiment analysis using setfit

# load two datasets from csv files in dataset dictionary
dataset = load_dataset('csv', data_files= {
    "train": "setfit-dataset-trainOLD.csv",
    "test": "setfit-dataset-testOLD.csv"
})

# Encode the sentiment labels (positive / negative) in both datasets
# Ex. positive becomes "1" and negative becomes "0" - makes the data "readable" for the model
# partially taken from https://hackernoon.com/mastering-few-shot-learning-with-setfit-for-text-classification
le = LabelEncoder()

sentiment_encodings_train = le.fit_transform(dataset["train"]['label'])
dataset["train"] = dataset["train"].remove_columns("label").add_column("label", sentiment_encodings_train).cast(dataset["train"].features)

sentiment_encodings_test = le.fit_transform(dataset["test"]['label'])
dataset["test"] = dataset["test"].remove_columns("label").add_column("label", sentiment_encodings_test).cast(dataset["test"].features)

# base pretrained model from SetFit library
model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# fine tune pretrained model using datasets
trainer = Trainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"]
)

# Train model
trainer.train()

# get accuracy of model against test data
metrics = trainer.evaluate()

print(metrics)



