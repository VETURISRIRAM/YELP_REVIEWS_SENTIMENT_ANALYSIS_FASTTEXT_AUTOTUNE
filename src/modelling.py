import fasttext

#TODO: Change these paths according to your local setup or you can use the relative paths.
TRAIN_FILE = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\fasttext_files\\train.txt"
TEST_FILE = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\fasttext_files\\test.txt"
VAL_FILE = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\data\\fasttext_files\\val.txt"
MODELS_DIR_PATH = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\models\\yelp_reviews_classifier.bin"

model = fasttext.train_supervised(input=TRAIN_FILE
                                  , autotuneValidationFile=VAL_FILE
                                  )

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

results = model.test(TEST_FILE)
model.save_model(MODELS_DIR_PATH)
print_results(*results)