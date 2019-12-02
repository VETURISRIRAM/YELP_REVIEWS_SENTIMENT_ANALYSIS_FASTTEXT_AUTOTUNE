import os
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

#TODO: Change these paths according to your local setup or you can use the relative paths.
MULTIPROC_FILES_PATH = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\preprocessed_multiproc_files\\"
TRAIN_FILE = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\fasttext_files\\train.txt"
TEST_FILE = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\fasttext_files\\test.txt"
VAL_FILE = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\fasttext_files\\val.txt"

def write_files(df):
    """
    Function to create and write the input fasttext files.
    :param df: Preprocessed DataFrame.
    """

    X = df["text"]
    y = df["sentiment"]
    print("X and Y Split Done: ", datetime.datetime.now())

    # Split into train-test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.1)

    # Split into train-val sets.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        stratify=y_train,
                                                        test_size=0.2)
    print("Stratified Split Done. Writing Files now: ", datetime.datetime.now())

    # Write the train file.
    with open(TRAIN_FILE, "w") as train_file_handler:
        for X_train_entry, y_train_entry in zip(X_train, y_train):
            line_to_write = "__label__" + str(y_train_entry) + "\t" + str(X_train_entry) + "\n"
            try:
                train_file_handler.write(line_to_write)
            except:
                print(line_to_write)
                break
    print("Train File Written: ", datetime.datetime.now())

    # Write the test file.
    with open(TEST_FILE, "w") as test_file_handler:
        for X_test_entry, y_test_entry in zip(X_test, y_test):
            line_to_write = "__label__" + str(y_test_entry) + "\t" + str(X_test_entry) + "\n"
            try:
                test_file_handler.write(line_to_write)
            except:
                print(line_to_write)
                break
    print("Test File Written: ", datetime.datetime.now())

    # Write the validation file for FastText Autotune.
    with open(VAL_FILE, "w") as val_file_handler:
        for X_val_entry, y_val_entry in zip(X_val, y_val):
            line_to_write = "__label__" + str(y_val_entry) + "\t" + str(X_val_entry) + "\n"
            try:
                val_file_handler.write(line_to_write)
            except:
                print(line_to_write)
                break
    print("Val File Written: ", datetime.datetime.now())


def combine_multiproc_files():
    """
    Function to combine the files created by multiprocessing.
    :return:
    """

    temp_df = pd.read_csv(MULTIPROC_FILES_PATH + "yelp_multiproc_file_0.csv", nrows=4)
    columns = temp_df.columns
    df = pd.DataFrame(columns=columns)
    for file in os.listdir(MULTIPROC_FILES_PATH):
        if file.endswith(".csv"):
            print(f"Now, concatinating {file}.")
            ind_df = pd.read_csv(os.path.join(MULTIPROC_FILES_PATH, file))
            df = pd.concat([df, ind_df], axis=0)
    df = df.loc[:, ~df.columns.str.match('Unnamed')]

    return df


if __name__ == "__main__":

    print("Started: ", datetime.datetime.now())
    df = combine_multiproc_files()
    print("Combined Multiprocessed DataFrames: ", datetime.datetime.now())

    write_files(df)
    print("Done")
