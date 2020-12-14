import pickle

# To load the LinearSVC model with pickle
with open("model.pckl", "rb") as f:
       while True:
           try:
               model = pickle.load(f)
           except EOFError:
               break
