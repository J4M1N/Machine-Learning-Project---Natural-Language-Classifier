import pickle

with open('train_model.pkl', 'rb') as f:
    m = pickle.load(f)

with open('vectorizer.pkl','rb') as f:
    v = pickle.load(f)

input = "I am unhappy today"
vectorized_input = v.transform([input])
output = m.predict(vectorized_input)
print(output)
