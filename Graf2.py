import matplotlib.pyplot  as plt

x100RF=[10,20,30]
y100RF=[0.316,0.340,0.404]
y100RFTrain=[0.992,0.996,0.995]

x200RF=[10,20,30]
y200RF=[0.388,0.464,0.508]
y200RFTrain=[0.996,0.996,0.996]

fig=plt.figure()

ax1=fig.add_subplot(111)

ax1.scatter(x100RF,y100RFTrain,label='100x100 Window TRAIN',color='xkcd:magenta')
ax1.plot(x100RF,y100RFTrain,color='xkcd:magenta')
ax1.scatter(x100RF,y100RF,label='100x100 Window Test',color='xkcd:pink')
ax1.plot(x100RF,y100RF,color='xkcd:pink')

ax1.scatter(x200RF,y200RFTrain,label='100x100 Window TRAIN',color='xkcd:teal')
ax1.plot(x200RF,y200RFTrain,color='xkcd:teal')
ax1.scatter(x200RF,y200RF,label='200x200 Window Test',color='xkcd:cyan')
ax1.plot(x200RF,y200RF,color='xkcd:cyan')

plt.title('Random Forest ACA(#trainIm)')
plt.xlabel('Number of training images')
plt.ylabel('Average Classification Accuracy')

plt.legend(loc='upper left')
plt.savefig('ACA-RF')
plt.show()

