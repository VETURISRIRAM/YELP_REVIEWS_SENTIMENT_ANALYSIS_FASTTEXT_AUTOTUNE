import re
import datetime
import numpy as np
import pandas as pd
import multiprocessing

# Number of processes to spin up.
NUM_PROCESSES = multiprocessing.cpu_count() - 1

#TODO: Change these paths according to your local setup or you can use the relative paths.
DATA_PATH = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\yelp_review\\yelp_review.csv"
MULTIPROC_FILES_PATH = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\preprocessed_multiproc_files\\"


def get_sentiment(star):
    """
    Function to return the sentiment from the stars given.
    :param star: Number of stars.
    :return sentiment: Sentiment based on stars.
    """

    if star == 3:
        return "neutral"
    elif star < 3:
        return "negative"
    else:
        return "positive"


def preprocess_review(review):
    """
    Function to preprocess the text review.
    :param review: Review.
    :return preprocessed_review: Preprocessed Review.
    """

    # Minor preprocessing.
    review = review.lower().strip()
    review = review.lstrip(",").rstrip(",").replace("\n", " ").replace("\t", " ")
    review = re.sub('[^.,a-zA-Z0-9 \n\.]', '', review)
    preprocced_review = review
    #TODO Add code to handle special chars by regex.

    # N - Gramming.
    # n = 4
    # review = review.replace('^"', '').replace('"$', '').replace('""', '"')
    # review = "^" + review.replace(" ", "*") + "$" # Padding for short strings.
    # preprocced_review = " ".join([review[i: i + n] for i in range(len(review) - n + 1)])

    return preprocced_review


def preprocess_data(df, f_ind):
    """
    Function to preprocess the data.
    :param df: Raw DataFrame.
    :return df: Preprocessed DataFrame.
    """

    df = df[["stars", "text"]]
    df = df.dropna()
    print("Dropped Missing Rows: ", datetime.datetime.now())
    df["sentiment"] = df["stars"].apply(lambda star: get_sentiment(star))
    print("Label Processed:", datetime.datetime.now())
    df = df.drop(["stars"], axis=1)
    print("Dropped Stars Column: ", datetime.datetime.now())
    df["text"] = df["text"].apply(lambda review: preprocess_review(review))
    print("Preprocessed Reviews (String Manipulations and N-Gramming): ", datetime.datetime.now(), df.shape)
    df.to_csv(MULTIPROC_FILES_PATH + "yelp_multiproc_file_{}.csv".format(f_ind))


# Main Function starts here.
if __name__ == "__main__":

    print("Started: ", datetime.datetime.now())
    df = pd.read_csv(DATA_PATH)
    print("Data Read: ", datetime.datetime.now(), df.shape)

    dfs = np.array_split(df, NUM_PROCESSES)
    jobs = []
    for i in range(NUM_PROCESSES):
        print("Process {} started".format(i))
        p = multiprocessing.Process(target=preprocess_data(dfs[i], i))
        jobs.append(p)
        p.start()
