
import pickle

with open(f'membership_classifier.pkl', 'rb') as f:
    loaded = pickle.load(f)



print(type(loaded))
print(loaded.keys())  # this will show you the r

# loaded is the whole dict above
# loaded['model'] gets just the LogisticRegression object
model = loaded['classifier']


# Check the model loaded correctly
print(type(model))           # should show LogisticRegression
print(model.coef_)           # the learned coefficients — only exists if model was trained
print(model.intercept_)      # the intercept — same, only exists if trained
print(model.classes_)        # the classes it was trained on, e.g. [0, 1]
