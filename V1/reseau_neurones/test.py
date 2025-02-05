from keras import *

model = Sequential()

# Couche entrée
model.add(layers.Dense(units=3, input_shape=[1]))
# Couche cachée
model.add(layers.Dense(units=64))
# Couche sortie
model.add(layers.Dense(units=1))

entree = [1,2,3,4,5]
sortie = [2,4,6,8,10]

model.compile(loss='mean_squarred_error', optimizer='adam')

model.fit(x=entree, y=sortie, epochs=100)

while True :
    x= int(input("Nombre :"))
    print("prediction :" + str(model.predict([x])))