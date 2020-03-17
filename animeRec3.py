import pandas as pd

from sklearn.model_selection import KFold

import numpy as np
from keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from keras.models import Model
from keras import backend
from statistics import mean



#  Collaborative filtering,  word Embedding dense vector representation - with binary classification  NeuralNetwork regression model


dataset = pd.read_csv('rating.csv',nrows=2000)
dataset = dataset.sample(frac=1)  # shuffle data
dataset = dataset.loc[dataset['rating'] != -1] # remove unwanted data (done after meta-data exploration)
#dataset = dataset.loc[lambda x: x['rating'] >= 0]
dataset.drop_duplicates() # remove duplicate rows
num_of_rows = dataset.shape[0] # number of rows

index = 0
userIdOneHot = {}
for x,value in dataset['user_id'].items(): # user id dict
    if value not in userIdOneHot:
        userIdOneHot[value] = index
        index += 1
recipeOneHotIndex = 0
ids_to_idx = {}
idx_to_ids = []
for x,val in dataset['anime_id'].items(): # anime ids to index , index to anime ids
    idx_to_ids.append(val)
    if val not in ids_to_idx:
        ids_to_idx[val] = recipeOneHotIndex
        recipeOneHotIndex += 1


def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + backend.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))


user_id = dataset['user_id'].map(userIdOneHot.get).copy()

anime_id = dataset['anime_id'].map(ids_to_idx.get).copy()
rating = dataset['rating'].copy()
n_users = len(dataset.user_id.unique()) # size of user embedding - num of unique user ids
n_anime = len(dataset.anime_id.unique()) # size of anime embedding - num of unique anime ids

#print("---------------train-----shapes---------test-----------")

print("num of users:",n_users)
print("num of n_recipe:",n_anime)


print("----------------------------------------")

# First Emedding Layer for animes
anime_input = Input(shape=[1], name="anime-Input")
anime_embedding = Embedding(n_anime, 5, name="anime-Embedding")(anime_input)
anime_vec = Flatten(name="Flatten-anime")(anime_embedding)

# Second Emedding Layer,parallel to the first one, for users
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)
#print("------------------conc----------------------")
conc = Concatenate()([user_vec,anime_vec]) # concatenate the two embedding vectors
#print(conc)

# NetWork Dense layers with relu activation
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)
#model = Model([user_input, book_input], prod)
model = Model([user_input, anime_input], out)
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy',rmse,'mae',f1])

# K - Fold , n_splits is equal to num of rows --> Leave one Out
kf = KFold(n_splits=num_of_rows, shuffle=False)
accuracy_model = []
for train_index, test_index in kf.split(user_id):
    # Split train-test
    X1_train, X1_test = user_id.iloc[train_index], user_id.iloc[test_index]
    X2_train, X2_test = anime_id.iloc[train_index], anime_id.iloc[test_index]
    y_train, y_test = rating.iloc[train_index], rating.iloc[test_index]


    # print("----------------------------------------")
    #
    # print("YES:",y_train)
    # print("----------------------------------------")


    # Train the model
    history = model.fit([X1_train,X2_train], y_train, epochs=10, verbose=1)
    accuracy_model.append(model.evaluate([X1_test,X2_test],y_test ,batch_size=2,))
print(model.metrics_names)
print(accuracy_model)
print(*map(mean,zip(*accuracy_model)))
model.save('regression_model.h5')
anime_data = np.array(list(set(dataset.anime_id)))
anime_data = []
temp = anime_id.tolist()
for k in range(10):
   # print(anime_id[k])
    anime_data.append(temp[k*4])

anime_data = np.array(anime_data)
print("------------------anime_data----------------------")
print(anime_data) # anime ids
print("------------------user_data----------------------")
user = np.array([1,1,1,1,1,1,1,1,1,1]) # user id
print(user)
queryToPredict = [user,anime_data] # prediction for (user,animeId) for each tuple returns predicted rating
predictions = model.predict(queryToPredict)
print("------------------predictions----------------------")

print(predictions)


predictions = np.array([a[0] for a in predictions])
recommended_anime_ids = (-predictions).argsort()[:5]
print("Top 5:")
print(recommended_anime_ids , "real ID:" , idx_to_ids)
def realName(id):
    animeData=pd.read_csv('anime.csv')
    animeData = animeData.loc[animeData['anime_id'] == id]
    name = animeData['name']
    return name
for num in recommended_anime_ids:
    print(num," RealID:" , idx_to_ids[num],' ',realName(idx_to_ids[num]))

print("------------------recommended_anime_ids----------------------")

#print(recommended_recipe_ids)
# print(predictions[recommended_recipe_ids])



#print(recipe[recipe['id'].isin(recommended_recipe_ids)])