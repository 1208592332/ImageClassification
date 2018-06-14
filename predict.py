from keras.models import load_model
from utils import load_data
from sklearn.metrics import accuracy_score
import os
#data path
path_test = "../data2/test/"
dirlist_test = os.listdir(path_test)

params = {"norm_size":64}
test_X, test_y, test_X_name = load_data(path_test, dirlist_test, params)

model = load_model("../models/resnet_18_best.h5")
pred = model.predict(test_X)

right = 0
# for i in range(len(pred)):
#     if  test_y[i] == pred[i]:
#         right +=1
#     else:
#         print(test_X_name[i])
# print("Score: ",right / len(pred))

print(test_y)
print(pred)



