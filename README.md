# Yelp Reviews Sentiment Analysis using FastText

## Description

This project aims to classify the <a href="https://www.kaggle.com/yelp-dataset/yelp-dataset/version/4/">Kaggle Yelp review</a> in three classes.

1) Positive (If the stars are above 3).
2) Neutral  (If the stars are equal to 3).
3) Negative (If the stars are below 3).

## FastText Introduction

FastText as a library for efficient learning of word representations and sentence classification. It is written in C++ and supports multiprocessing during training. FastText allows you to train supervised and unsupervised representations of words and sentences. These representations (embeddings) can be used for numerous applications from data compression, as features into additional models, for candidate selection, or as initializers for transfer learning.

## Get the Data.

The date used in this project from the initial Kaggle dataset to the intermediate FastText files created could be downloaded from here.

In this repository, I have kept the `./data/` directory empty. You can place the downloaded folder (extracted) in the `./data/` and follow the below instructions.


## How to run?

Setup the project. I used the latest FastText from the <a href="https://github.com/facebookresearch/fastText">GitHub</a>.

I wanted to explore the `AutoTune` feature of FastText which enables the automatic Hyperparameter tuning. Using `AutoTune` feature, the model is trained with the best possible hyperparameters. According to my understanding, it is somewhat similar to the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">Sklearn's GridSearchCV</a> module.

The below script reads the data, creates the labels, does some minor text preprocessing using `multiprocessing`.

```
python preprocess_data.py
```

After the preprocessing is done, train-val-test files are to be created for the FastText model.

The format required by FastText is like `__label__positive    The restaurant was great.`

Notice the `__label__`. It's how FastText understands that `positive` is the label for the data `The restaurant was great.`

They could either be separated by a space or a tab.

The file extension does not matter. It could be any of the TXT/TSV/CSV or other extensions which can hold textual data.

The below command creates input files for FastText as described above.

```
python create_files.py
```

Now that we have the input files, we can go ahead and start training our classifier.

Model Training and Testing are fairly simple in FastText.

I am using the `AutoTune` functionality to tune the hyperparameters of my model. It can be set by passing the validation file to the `autotuneValidationFile` argument when you initiate the training.

The bin model is trained with the best hyperparameters and saved in the `./models/` directory. 

```
python modelling.py
```

Evaluation Results on the Test Set.

```
N	526167
P@1	0.863
R@1	0.863
```

You can also get results /predictions for your test review by running the below file. 

```
python test_model.py
```

## Example Testing

I gave some test inputs to the model and got the predictions as follows. 

```
print(model.predict("the food was really great"))
print(model.predict("the restaurant was horrible"))
print(model.predict("the salon was okay. Not bad!"))
```

Output/Predictions:

```
(('__label__positive',), array([0.99909163]))
(('__label__negative',), array([1.00000417]))
(('__label__neutral',), array([0.99479502]))
```

## References and Sources

Thanks to the authors of these articles!

1) <a href="https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3">FastText: Under the Hood</a> (Medium Article).
2) <a href="https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/">Python for NLP: Working with Facebook FastText Library</a> (StackAbuse Article).
3) <a href="https://fasttext.cc/">FastText Official Documentation</a> 
