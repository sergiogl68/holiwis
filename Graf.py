import matplotlib.pyplot  as plt

x100KNN=[10,20,30]
y100KNN=[0.284,0.372,0.388]

x200KNN=[10,20,30]
y200KNN=[0.352,0.42,0.464]

fig=plt.figure()

ax1=fig.add_subplot(111)

ax1.scatter(x100KNN,y100KNN,label='100x100 Window')
ax1.plot(x100KNN,y100KNN)
ax1.scatter(x200KNN,y200KNN,label='200x200 Window')
ax1.plot(x200KNN,y200KNN)

plt.title('K-nearest neighbor ACA(#trainIm)')
plt.xlabel('Number of training images')
plt.ylabel('Average Classification Accuracy')

plt.legend(loc='upper left')
plt.savefig('ACA-KNN')
plt.show()

