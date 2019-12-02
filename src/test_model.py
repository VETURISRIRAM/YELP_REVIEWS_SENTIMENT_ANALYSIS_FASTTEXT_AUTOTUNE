import fasttext

#TODO: Change these paths according to your local setup or you can use the relative paths.
MODELS_DIR_PATH = "C:\\Users\\srira\\Desktop\\Sriram\\Projects\\Yelp_Reviews\\models\\yelp_reviews_classifier.bin"

model = fasttext.load_model(MODELS_DIR_PATH)

print(model.predict("the food was really great"))
print(model.predict("the restaurant was horrible"))
print(model.predict("the salon was okay. Not bad!"))

# (('__label__positive',), array([0.99909163]))
# (('__label__negative',), array([1.00000417]))
# (('__label__neutral',), array([0.99479502]))
