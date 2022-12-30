import pickle
from build_vocab import Vocabulary
# open a file, where you stored the pickled data
file = open('/home/usr/Workspaces/tuanns-ai/NLP/cnn_lstm/data/vocab.pkl', 'rb')
vocab = ['tuan','tung','thanh']
vocab = Vocabulary()
print(vocab('thanh'))
# dump information to that file
# data = pickle.load(file)

# # close the file
# file.close()

# print('Showing the pickled data:')
# print(data)
# # cnt = 0
# for item in data:
#     print('The data ', cnt, ' is : ', item)
#     cnt += 1