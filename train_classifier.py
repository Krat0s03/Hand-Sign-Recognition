import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np


data_dict = pickle.load(open('./data_words.pickle', 'rb'))
# for i, seq in enumerate(data_dict['data']):
#     if len(seq)==84:
#         print(f"Sequence {i} length: {len(seq)}")
# data = pad_sequences(data_dict['data'], padding='post', truncating='post', maxlen=42)
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model_words.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
