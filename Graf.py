import matplotlib.pyplot  as plt

x100KNN=[10,20,30]
y100KNN=[0.284,0.372,0.388]
y100KNNTrain=[0.524,0.580,0.608]
x200KNN=[10,20,30]
y200KNN=[0.352,0.42,0.464]
y200KNNTrain=[0.540,0.610,0.654]

fig=plt.figure()

ax1=fig.add_subplot(111)


ax1.scatter(x100KNN,y100KNNTrain,label='100x100 Window TRAIN', color='xkcd:green')
ax1.plot(x100KNN,y100KNNTrain, color='xkcd:green')
ax1.scatter(x100KNN,y100KNN,label='100x100 Window Test', color='xkcd:lime green')
ax1.plot(x100KNN,y100KNN, color='xkcd:lime green')

ax1.scatter(x200KNN,y200KNNTrain,label='200x200 Window TRAIN',color='xkcd:dark blue')
ax1.plot(x200KNN,y200KNNTrain,color='xkcd:dark blue')
ax1.scatter(x200KNN,y200KNN,label='200x200 Window Test',color='xkcd:azure')
ax1.plot(x200KNN,y200KNN,color='xkcd:azure')


plt.title('K-nearest neighbor ACA(#trainIm)')
plt.xlabel('Number of training images')
plt.ylabel('Average Classification Accuracy')

plt.legend(loc='upper left')
plt.savefig('ACA-KNN')
plt.show()

